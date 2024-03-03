import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

# Import the base VisionTransformer class from a local module.
from .MAE import VisionTransformer

class LUM_ViT(VisionTransformer):
    def __init__(self, target_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = self.patch_embed.num_patches  # Number of patches into which the input image is divided.
        self.target_rate = target_rate  # The target rate of tokens to be masked.

        # Initialize the mask, which will handle the masking of the patches.
        self.token_mask = TokenChannelMask(num_patches=self.num_patches, embed_dim=self.embed_dim,
                                           target_rate=self.target_rate)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, mask = self.token_mask(x)  # Apply token masking to the embedded patches.

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        if self.dist_token is None:
            if self.global_pool:
                return self.pre_logits(x), mask
            else:
                return self.pre_logits(x[:, 0]), mask
        else:
            return x[:, 0], x[:, 1], mask

    def forward(self, x):
        x, mask = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist, mask
            else:
                return (x + x_dist) / 2, mask
        else:
            x = self.head(x)
        if self.training and not self.random_mask:
            return x, mask
        else:
            return x

# Define the learnable mask.
class TokenChannelMask(nn.Module):
    # Initialize the mask with the number of patches, embedding dimension, and other options.
    def __init__(self, num_patches=196, embed_dim=768, make_learnable_token=True, norm_layer=nn.LayerNorm,
                latter=False, target_rate=0.1):
        super().__init__()
        self.target_rate = target_rate  # The target rate for masking.
        self.num_patches = num_patches  # Number of patches.
        self.embed_dim = embed_dim  # Embedding dimension of the patches.
        self.make_learnable_token = make_learnable_token  # Whether to make the token learnable.
        self.latter = latter  # A flag that seems to be used for conditional behavior in masking.
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()  # Normalization layer.

        # If not in 'latter' mode, initialize parameters for the mask and a linear layer for patch masking.
        if not latter:
            self.patch_mask_para = nn.Parameter(torch.ones((self.num_patches, self.embed_dim, 2)), requires_grad=True)
            self.patch_fc = nn.Linear(self.num_patches*2, self.num_patches*2)
            trunc_normal_(self.patch_mask_para, std=0.02)
            self.toval = nn.LogSoftmax(dim=-1)  # LogSoftmax for converting mask values.

        # If learnable tokens are enabled, initialize them.
        if self.make_learnable_token:
            self.learnable_token = nn.Parameter(torch.zeros((1, self.num_patches, self.embed_dim)), requires_grad=True)
            trunc_normal_(self.learnable_token, std=0.02)

    def forward(self, x):
        B, N, C = x.shape  # Unpack batch size, number of patches, and channel dimension.
        assert N == self.num_patches and C == self.embed_dim

        self.get_mask(B)  # Compute the mask for the current batch size.
        x *= self.mask  # Apply the mask to the input.

        # If learnable token is enabled, replace masked positions with the learnable token.
        if self.make_learnable_token:
            mask_pos = (self.mask == 0)
            x = torch.where(mask_pos.expand_as(x), self.learnable_token.expand_as(x).to(x), x)

        x = self.norm_layer(x)  # Apply normalization.

        return x, self.mask  # Return the masked input and the mask itself.

    def get_mask(self, batch_size):
        # If in 'latter' mode, return the precomputed mask.
        if self.latter:
            return self.mask

        # Calculate mask values and apply LogSoftmax.
        patch_tmp = (self.patch_fc(self.patch_mask_para.permute(1, 0, 2).reshape(-1, self.num_patches*2))
                     .reshape(-1, self.num_patches, 2).permute(1, 0, 2))
        self.patch_mask_val = self.toval(patch_tmp).unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Depending on whether it's training mode and if gradients are required, compute the mask accordingly.
        if self.training and self.patch_mask_para.requires_grad:
            # Apply Gumbel softmax to get a hard mask.
            self.patch_mask = F.gumbel_softmax(self.patch_mask_val, hard=True)[:, :, :, 0]
            self.mask = self.patch_mask
        else:
            # Otherwise, use a soft mask based on the softmax values.
            patch_mask_val_softmax = F.softmax(self.patch_mask_val, dim=-1)[:, :, :, 0]
            mask_val_softmax = patch_mask_val_softmax
            num_keep = int(mask_val_softmax.numel() * self.target_rate)
            _, indices = torch.sort(mask_val_softmax.view(-1), descending=True)

            new_tensor = torch.zeros_like(mask_val_softmax)
            top_indices = indices[:num_keep]
            top_indices = unravel_index(top_indices, mask_val_softmax.shape)
            new_tensor[top_indices] = 1
            self.mask = new_tensor

        return self.mask  # Return the mask.

    # A method to load the mask from a dictionary of model parameters.
    def load_mask(self, model_dict):
        missed_names = []
        # Iterate over the parameters and match with keys in the provided dictionary.
        for name, para in self.named_parameters():
            miss_flag = True
            for key, dict_para in model_dict.items():
                if key.endswith(name):
                    miss_flag = False
                    para.data.copy_(dict_para.data)
            if miss_flag:
                missed_names.append(name)
        return missed_names  # Return names of parameters that weren't found in the dictionary.

    # A method to provide a string representation of the module.
    def extra_repr(self) -> str:
        return f'num_patches = {self.num_patches}, embed_dim = {self.embed_dim}, ' \
               f'make_learnable_token= {self.make_learnable_token}, norm_layer = {self.norm_layer}'

# A utility function to unravel a flat index into indices for a multidimensional shape.
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
