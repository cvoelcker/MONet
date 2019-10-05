import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def create_coord_buffer(patch_shape):
    ys = torch.linspace(-1, 1, patch_shape[0])
    xs = torch.linspace(-1, 1, patch_shape[1])
    xv, yv = torch.meshgrid(ys, xs)
    coord_map = torch.stack((xv, yv)).unsqueeze(0)
    return coord_map


def alternate_inverse(theta):
    inv_theta = torch.zeros_like(theta)
    inv_theta[:, 0, 0] = 1 / theta[:, 0, 0]
    inv_theta[:, 1, 1] = 1 / theta[:, 1, 1]
    inv_theta[:, 0, 2] = -theta[:, 0, 2] / theta[:, 0, 0]
    inv_theta[:, 1, 2] = -theta[:, 1, 2] / theta[:, 1, 1]
    return inv_theta


def invert(x, theta, image_shape, padding='zeros'):
    inverse_theta = alternate_inverse(theta)
    if x.size()[1] == 1:
        size = torch.Size((x.size()[0], 1, *image_shape))
    elif x.size()[1] == 3:
        size = torch.Size((x.size()[0], 3, *image_shape))
    grid = F.affine_grid(inverse_theta, size)
    x = F.grid_sample(x, grid, padding_mode=padding)

    return x


def differentiable_sampling(mean, sigma, prior_sigma, reducer='mean'):
    dist = dists.Normal(mean, sigma)
    dist_0 = dists.Normal(0., prior_sigma)
    z = mean + sigma * dist_0.sample()
    kl_z = dists.kl_divergence(dist, dist_0)
    if reducer == 'mean':
        kl_z = torch.mean(kl_z, [1])
    elif reducer == 'sum':
        kl_z = torch.sum(kl_z, [1])
    else:
        raise NotImplementedError('Reducer must be sum or mean')
    return z, kl_z


def reconstruction_likelihood(x, recon, mask, sigma, i=None):
    dist = dists.Normal(x, sigma)
    p_x = dist.log_prob(recon) * mask
    return p_x


def kl_mask(mask_pred, mask, reducer='mean'):
    tr_masks = mask.view(mask.size()[0], -1)
    tr_mask_preds = mask_pred.view(mask_pred.size()[0], -1)

    q_masks = dists.Bernoulli(probs=tr_masks)
    q_masks_recon = dists.Bernoulli(probs=tr_mask_preds)
    kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
    if reducer == 'mean':
        kl_masks = torch.mean(kl_masks, [1])
    elif reducer == 'sum':
        kl_masks = torch.sum(kl_masks, [1])
    else:
        raise NotImplementedError('Reducer must be sum or mean')
    return kl_masks


def transform(x, grid, theta):
    x = F.grid_sample(x, grid)
    return x


def center_of_mass(mask, device='cuda'):
    grids = [torch.Tensor(grid).to(device) for grid in np.ogrid[[slice(0, i) for i in mask.shape[-2:]]]]
    norm = torch.sum(mask, [-2, -1])
    return torch.stack([torch.sum(mask * grids[d], [-2, -1]) / norm for d in range(2)], -1) 


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2 ** i))
            cur_in_channels = channel_base * 2 ** i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks - 1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2 ** i,
                                                  channel_base * 2 ** (i - 1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks - 2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2 ** (i + 1),
                                             channel_base * 2 ** i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks - 1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i - 1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)
