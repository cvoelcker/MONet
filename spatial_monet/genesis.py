import collections
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

import spatial_monet.util.network_util as net_util
import spatial_monet.util.sylvester_layers as sylvester


class SylvesterConvEncoder(nn.Module):
    def __init__(self, img_shape=(32, 32), latent_dim=32, input_size=4, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_shape = img_shape
        self.conv_layers = nn.Sequential(
            sylvester.GatedConv2d(input_size,  32,  5, 1, 2),
            sylvester.GatedConv2d(32, 32,  5, 2, 2),
            sylvester.GatedConv2d(32, 64,  5, 1, 2),
            sylvester.GatedConv2d(64, 64,  5, 2, 2),
            sylvester.GatedConv2d(64, 64,  5, 1, 2),
            sylvester.GatedConv2d(64, 256, 16, 1, 0))
        self.mlp = nn.Sequential(
                nn.ELU(),
                nn.Linear(256, latent_dim * 2))

    def forward(self, x):
        conv = self.conv_layers(x)
        conv = conv.view(x.shape[0], -1)
        return self.mlp(conv)


class SylvesterConvDecoder(nn.Module):
    def __init__(self, img_shape=(32, 32), latent_dim=32, output_size=1, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        network = [
            sylvester.GatedConvTranspose2d(latent_dim, 64, 16, 1, 0),
            sylvester.GatedConvTranspose2d(64, 64, 5, 1, 2),
            sylvester.GatedConvTranspose2d(64, 32, 5, 2, 2, output_padding = 1),
            sylvester.GatedConvTranspose2d(32, 32, 5, 1, 2),
            sylvester.GatedConvTranspose2d(32, 32, 5, 2, 2, output_padding = 1),
            sylvester.GatedConvTranspose2d(32, 32, 5, 1, 2),
            nn.Conv2d(32, output_size, 1)
        ]
        if output_size > 1:
            # constrain pixels to [0, 1] interval
            network.append(nn.Sigmoid())

        self.network = nn.Sequential(*network)

    def forward(self, x):
        # adds coordinate information to z and 
        # produces a tiled representation of z
        result = self.network(x.view(-1, self.latent_dim, 1, 1))
        return result


class ConvEncoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(self, img_shape=(32, 32), latent_dim=32, input_size=4, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_shape = img_shape

        self.network = nn.Sequential(
            nn.Conv2d(input_size, 16, 3, padding=(1, 1)),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, padding=(1, 1)),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_size = int(
            64 * self.patch_shape[0] / (2 ** 3) * self.patch_shape[1] / (
                    2 ** 3))
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_size, 2 * self.latent_dim),
            nn.ELU())
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim))

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, self.conv_size)
        x = self.mlp(x)
        mean = self.mean_mlp(x[:, :self.latent_dim])
        sigma = F.softplus(self.sigma_mlp(x[:, self.latent_dim:])) + 1e-10
        return mean, sigma


class RecEncoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(self, img_shape=(32, 32), latent_dim=32, input_size=3, num_slots=10, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_slots = num_slots

        self.rec = nn.GRU(2 * self.latent_dim, 2 * self.latent_dim, batch_first=True)
        self.joint_mlp = nn.Sequential(
            nn.ELU(),
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
            nn.ELU(),
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim))
        self.mean_mlp = nn.Sequential(
            nn.ELU(),
            nn.Linear(2 * self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
            nn.ELU(),
            nn.Linear(2 * self.latent_dim, self.latent_dim))

    def forward(self, x):
        latent_dim = self.latent_dim
        num_slots = self.num_slots
        # reshaping is necessary for different scalings of conv and rec part
        x_rec, _ = self.rec(x)
        x_rec = x_rec.contiguous().view(-1, 2 * latent_dim)
        x_rec = self.joint_mlp(x_rec)
        mean = self.mean_mlp(x_rec)
        sigma = F.softplus(self.sigma_mlp(x_rec)) + 1e-10
        mean = mean.view(-1, num_slots, latent_dim)
        sigma = sigma.view(-1, num_slots, latent_dim)
        return mean, sigma


class DecoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    Implements a spatial braodcast decoder architecture
    """

    def __init__(self, img_shape=(32, 32), latent_dim=32, output_size=1, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        network = [
            nn.Conv2d(self.latent_dim + 2, 64, 3, padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 32, 3, padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(32, output_size, 1),
        ]
        if output_size > 1:
            # constrain pixels to [0, 1] interval
            network.append(nn.Sigmoid())

        self.network = nn.Sequential(*network)

        # coordinate patching trick
        coord_map = net_util.create_coord_buffer(self.img_shape)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, x):
        # adds coordinate information to z and 
        # produces a tiled representation of z
        z_scaled = x.unsqueeze(-1).unsqueeze(-1)
        z_tiled = z_scaled.repeat(1, 1, self.img_shape[0],
                                  self.img_shape[1])
        coord_map = self.coord_map_const.repeat(x.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.network(inp)
        return result


class GENESIS(nn.Module):
    """
    Full model for image reconstruction and mask prediction
    """

    def __init__(
            self, bg_sigma=0.01, fg_sigma=0.05, latent_prior=1.,
            latent_dim=16, patch_shape=(16, 16),
            image_shape=(256, 256), num_blocks=2, num_slots=8,
            constrain_theta=False, beta=1., gamma=1., 
            softmax_masks=False, use_sylvester=True, 
            debug=False, **kwargs):
        super().__init__()
        self.sigma = bg_sigma
        self.latent_prior = latent_prior
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.num_slots = num_slots
        self.softmax_masks = softmax_masks
        self.debug = debug
        self.counter = 0

        # TODO: currently unused, replace with GECO
        self.beta = beta
        self.gamma = gamma
        
        # networks
        # to encode the masks
        if use_sylvester:
            self.mask_encoder = SylvesterConvEncoder(
                    img_shape=self.image_shape, 
                    latent_dim=self.latent_dim,
                    num_slots=num_slots,
                    input_size=3)
            self.mask_decoder = SylvesterConvDecoder(
                    img_shape=self.image_shape,
                    latent_dim=self.latent_dim,
                    output_size=1)
        else:
            self.mask_encoder = ConvEncoderNet(
                    img_shape=self.image_shape, 
                    latent_dim=self.latent_dim,
                    num_slots=num_slots,
                    input_size=3)
            self.mask_decoder = DecoderNet(
                    img_shape=self.image_shape,
                    latent_dim=self.latent_dim,
                    output_size=1)
        self.rec_encoder = RecEncoderNet(
                img_shape=self.image_shape, 
                latent_dim=self.latent_dim,
                num_slots=num_slots)

        # to encode the images
        self.img_encoder = ConvEncoderNet(
                img_shape=self.image_shape, 
                latent_dim=self.latent_dim,
                input_size=4)
        self.img_decoder = DecoderNet(
                img_shape=self.image_shape,
                latent_dim=self.latent_dim,
                output_size=3)
        
        # to encode the prior means
        self.rec_prior = nn.GRU(self.latent_dim, self.latent_dim, batch_first=True)
        self.mlp_prior = nn.Sequential(
                nn.Linear(self.latent_dim, 2 * self.latent_dim),
                nn.ELU(),
                nn.Linear(2 * self.latent_dim, self.latent_dim))
        
        # graph_depth = grid_x, grid_y, z_mask_mean/std, z_img_mean/std
        self.graph_depth = 2 + 4 * latent_dim
        self.graph_size = (num_slots, num_slots)

    def forward(self, x):
        """
        Main model forward pass
        """
        batch_size = x.shape[0]
        if self.debug and self.counter > 500:
            from IPython.core.debugger import Tracer; Tracer()() 
        self.counter += 1

        # generate recurrent mask z_s and logits
        img_enc = self.mask_encoder(x)
        mask_mean, mask_std = self.rec_encoder(img_enc.unsqueeze(1).repeat(1, self.num_slots, 1))
        mask_mean = mask_mean.view(batch_size * self.num_slots, -1)
        mask_std = mask_std.view(batch_size * self.num_slots, -1)
        z_mask = dists.Normal(mask_mean, mask_std).rsample()
        mask_logits = self.mask_decoder(z_mask)
        mask_logits = mask_logits.view(batch_size, self.num_slots, 1, *self.image_shape)

        # generate masks via softmax or via looped stick breaking
        if self.softmax_masks:
            masks = torch.softmax(mask_logits, 1)
        else:
            masks = torch.sigmoid(mask_logits)
            scope = torch.ones_like(masks[:, 0])
            final_masks = []
            for i in range(self.num_slots-1):
                final_masks.append(scope * masks[:, i])
                scope = scope * (1 - masks[:, i])
            final_masks.append(scope)
            masks = torch.stack(final_masks, 1)
        masks = masks.view(-1, 1, *self.image_shape)
        reconstruction_input = x.repeat_interleave(self.num_slots, 0)

        # generate reconstructions
        reconstruction_input = torch.cat([reconstruction_input, masks], 1)
        recon_mean, recon_std = self.img_encoder(reconstruction_input)
        z_recons = dists.Normal(recon_mean, recon_std).rsample()

        recons = self.img_decoder(z_recons)

        # calculate reconstruction likelihood
        p_x = net_util.reconstruction_likelihood(reconstruction_input[:, :3], recons, masks, self.sigma)
        p_x = p_x.reshape(batch_size, self.num_slots, 3, *self.image_shape)

        # reshape all components for images
        z_mask = z_mask.view(batch_size, self.num_slots, -1)
        mask_mean = mask_mean.view(batch_size, self.num_slots, -1)
        mask_std = mask_std.view(batch_size, self.num_slots, -1)
        z_recons = z_recons.view(batch_size, self.num_slots, -1)
        recon_mean = recon_mean.view(batch_size, self.num_slots, -1)
        recon_std = recon_std.view(batch_size, self.num_slots, -1)

        masks = masks.view(batch_size, self.num_slots, 1, *self.image_shape)
        recons = recons.view(batch_size, self.num_slots, 3, *self.image_shape)

        # calculate priors and KL term
        _prior_u = torch.zeros_like(z_mask[:, :1, :])
        _prior_input = torch.cat([_prior_u, z_mask[:, :-1]], 1)
        prior_mask_mean, _ = self.rec_prior(_prior_input)
        prior_dist_mask = dists.Normal(prior_mask_mean, 1.)
        poster_dist_mask = dists.Normal(mask_mean, mask_std)
        kl_mask = dists.kl_divergence(poster_dist_mask, prior_dist_mask)

        prior_img_mean = self.mlp_prior(z_mask.view(batch_size * self.num_slots, -1))
        prior_img_mean = prior_img_mean.view(batch_size, self.num_slots, -1)
        prior_dist_img = dists.Normal(prior_img_mean, 1.)
        poster_dist_img = dists.Normal(recon_mean, recon_std)
        kl_recon = dists.kl_divergence(poster_dist_img, prior_dist_img)

        # reconstruct image
        img = torch.sum(masks * recons, 1)

        # reshape all loss terms
        kl_mask = kl_mask.view(batch_size, self.num_slots, -1).sum(-1)
        kl_recon = kl_recon.view(batch_size, self.num_slots, -1).sum(-1)

        total_loss = -p_x.sum([-4,-3,-2,-1]) + kl_mask.sum(-1) + kl_recon.sum(-1)

        # currently missing is the mask reconstruction loss
        return_dict = {'loss': total_loss,
                       'total_reconstruction': img,
                       'reconstructions': recons,
                       'reconstruction_loss': p_x,
                       'masks': masks,
                       'latents': z_recons,
                       'latents_mean': recon_mean,
                       'latents_std': recon_std,
                       'theta': z_mask,
                       'thetas_mean': mask_mean,
                       'thetas_std': mask_std,
                       'kl_loss': kl_mask + kl_recon}
        return return_dict

    def build_flat_image_representation(self, x, return_dists=False):
        batch_size = x.shape[0]

        img_enc = self.mask_encoder(x)
        mask_mean, mask_std = self.rec_encoder(img_enc.unsqueeze(0).repeat(1, self.num_slots, 1))
        mask_mean = mask_mean.view(batch_size * self.num_slots, -1)
        mask_std = mask_std.view(batch_size * self.num_slots, -1)
        z_mask = dists.Normal(mask_mean, mask_std).rsample()
        mask_logits = self.mask_decoder(z_mask)
        mask_logits = mask_logits.view(batch_size, self.num_slots, 1, *self.image_shape)

        z_mask = z_mask.reshape(batch_size, self.num_slots, -1)

        # generate masks via softmax (to forgo for loop for stick-breaking)
        if self.softmax_masks:
            masks = torch.softmax(mask_logits, 1)
        else:
            masks = torch.sigmoid(mask_logits)
            scope = torch.ones_like(masks[:, 0])
            final_masks = []
            for i in range(self.num_slots-1):
                final_masks.append(scope * masks[:, i])
                scope = scope * (1 - masks[:, i])
            final_masks.append(scope)
            masks = torch.stack(final_masks, 1)
        masks = masks.view(-1, 1, *self.image_shape)
        reconstruction_input = x.repeat_interleave(self.num_slots, 0)
        
        # generate reconstructions
        reconstruction_input = torch.cat([reconstruction_input, masks], 1)
        recon_mean, recon_std = self.img_encoder(reconstruction_input)

        mask_mean = mask_mean.view(batch_size, self.num_slots, -1)
        mask_std = mask_std.view(batch_size, self.num_slots, -1)
        recon_mean = recon_mean.view(batch_size, self.num_slots, -1)
        recon_std = recon_std.view(batch_size, self.num_slots, -1)

        masks = masks.view(batch_size, self.num_slots, *self.image_shape)

        # calculate priors and KL term
        _prior_u = torch.zeros_like(z_mask[:, :1, :])
        _prior_input = torch.cat([_prior_u, z_mask[:, :-1]], 1)
        prior_mask_mean, _ = self.rec_prior(_prior_input)
        prior_dist_mask = dists.Normal(prior_mask_mean, 1.)
        poster_dist_mask = dists.Normal(mask_mean, mask_std)
        kl_mask = dists.kl_divergence(poster_dist_mask, prior_dist_mask)

        prior_img_mean = self.mlp_prior(z_mask.view(batch_size * self.num_slots, -1))
        prior_img_mean = prior_img_mean.view(batch_size, self.num_slots, -1)
        prior_dist_img = dists.Normal(prior_img_mean, 1.)
        poster_dist_img = dists.Normal(recon_mean, recon_std)
        kl_recon = dists.kl_divergence(poster_dist_img, prior_dist_img)

        # offset necessary for dark masks
        grid = (net_util.center_of_mass(masks + 1e-20) - (self.image_shape[0]/2)) / self.image_shape[0]
        assert not torch.any(torch.isnan(grid))

        # construct the full latent per object representation
        output_mean  = torch.cat([grid, mask_mean, recon_mean], 2)
        output_std = torch.cat([mask_std, recon_std], 2)
        
        if return_dists:
            return output_mean, output_std, kl_mask + kl_recon
        else:
            return output_mean, kl_mask + kl_recon

    def reconstruct_from_latent(self, latent_mean, latent_std, imgs=None):
        """
        Given a latent representation of the image, construct a full image again

        Inputs:
            - x: torch.Tensor shapes (batch, num_slots, latent_dims + 6[theta]
            + 2[pos])
        """
        batch_size = latent_mean.shape[0]

        mask_mean = latent_mean[:, :, 2:self.latent_dim+2].view(-1, self.latent_dim)
        recon_mean = latent_mean[:, :, self.latent_dim+2:].view(-1, self.latent_dim)

        mask_std = latent_std[:, :, :self.latent_dim].view(-1, self.latent_dim)
        recon_std = latent_std[:, :, self.latent_dim:].view(-1, self.latent_dim)

        z_mask = dists.Normal(mask_mean, mask_std).rsample()
        mask_logits = self.mask_decoder(z_mask)
        mask_logits = mask_logits.view(batch_size, self.num_slots, 1, *self.image_shape)
        
        # generate masks via softmax (to forgo for loop for stick-breaking)
        masks = torch.softmax(mask_logits, 1)
        masks = masks.view(-1, 1, *self.image_shape)
        
        # generate reconstructions
        z_recons = dists.Normal(recon_mean, recon_std).rsample()

        recons = self.img_decoder(z_recons)

        # reshape all components for images
        z_mask = z_mask.view(batch_size, self.num_slots, -1)
        mask_mean = mask_mean.view(batch_size, self.num_slots, -1)
        mask_std = mask_std.view(batch_size, self.num_slots, -1)
        z_recons = z_recons.view(batch_size, self.num_slots, -1)
        recon_mean = recon_mean.view(batch_size, self.num_slots, -1)
        recon_std = recon_std.view(batch_size, self.num_slots, -1)

        # reconstruct image
        img = torch.sum(masks * recons, 1)
        
        # calculate priors and KL term
        _prior_u = torch.zeros_like(z_mask[:, :1, :])
        _prior_input = torch.cat([_prior_u, z_mask[:, :-1]], 1)
        prior_mask_mean, _ = self.rec_prior(_prior_input)
        prior_dist_mask = dists.Normal(prior_mask_mean, 1.)
        poster_dist_mask = dists.Normal(mask_mean, mask_std)
        kl_mask = dists.kl_divergence(poster_dist_mask, prior_dist_mask)

        prior_img_mean = self.mlp_prior(z_mask.view(batch_size * self.num_slots, -1))
        prior_img_mean = prior_img_mean.view(batch_size, self.num_slots, -1)
        prior_dist_img = dists.Normal(prior_img_mean, 1.)
        poster_dist_img = dists.Normal(recon_mean, recon_std)
        kl_recon = dists.kl_divergence(poster_dist_img, prior_dist_img)

        # reshape all loss terms
        kl_mask = kl_mask.view(batch_size, self.num_slots, -1).sum(-1)
        kl_recon = kl_recon.view(batch_size, self.num_slots, -1).sum(-1)

        # calculate reconstruction likelihood
        if imgs is not None:
            p_x = net_util.reconstruction_likelihood(
                imgs.repeat(self.num_slots, 1, 1, 1),
                recons,
                masks,
                self.sigma)
            p_x = p_x.view(batch_size, self.num_slots, -1).sum(-1)
            masks = masks.view(batch_size, self.num_slots, 1, *self.image_shape)
            recons = recons.view(batch_size, self.num_slots, 3, *self.image_shape)

            return p_x, kl_mask + kl_recon, img
        
        masks = masks.view(batch_size, self.num_slots, 1, *self.image_shape)
        recons = recons.view(batch_size, self.num_slots, 3, *self.image_shape)

        return kl_mask + kl_recon, img

