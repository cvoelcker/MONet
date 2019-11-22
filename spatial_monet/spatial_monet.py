import collections
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

import spatial_monet.util.network_util as net_util


class EncoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(self, latent_dim=32, patch_shape=(32, 32),
                 input_size=8,
                 **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_shape = patch_shape

        self.network = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_size = int(
            64 * self.patch_shape[0] / (2 ** 3) * self.patch_shape[1] / (
                    2 ** 3))
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_size, 2 * self.latent_dim),
            nn.ReLU(inplace=False))
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ELU(),
            nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ELU(),
            nn.Linear(self.latent_dim, self.latent_dim))

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, self.conv_size)
        x = self.mlp(x)
        mean = self.mean_mlp(x[:, :self.latent_dim])
        sigma = self.sigma_mlp(x[:, self.latent_dim:])
        return mean, sigma


class DecoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(self, latent_dim=32, patch_shape=(32, 32),
                 **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_shape = patch_shape

        # gave it inverted hourglass shape
        # maybe that helps (random try)
        self.network = nn.Sequential(
            nn.Conv2d(self.latent_dim + 2, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 5, padding=(2, 2)),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 4, 1),
        )

        # coordinate patching trick
        coord_map = net_util.create_coord_buffer(self.patch_shape)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, x):
        # adds coordinate information to z and 
        # produces a tiled representation of z
        z_scaled = x.unsqueeze(-1).unsqueeze(-1)
        z_tiled = z_scaled.repeat(1, 1, self.patch_shape[0],
                                  self.patch_shape[1])
        coord_map = self.coord_map_const.repeat(x.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.network(inp)
        return result


class SpatialLocalizationNet(nn.Module):
    """
    Attention network for object localization
    """

    def __init__(self, image_shape=(256, 256), patch_shape=(32, 32),
                 constrain_theta=False, num_slots=8, **kwargs):
        super().__init__()

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.constrain_theta = constrain_theta
        self.num_slots = num_slots

        self.min = 0.05
        self.max = 0.2

        self.detection_network = nn.Sequential(
            # block 1
            nn.Conv2d(8, 16, 3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            # output size = 32, width/2, length/2

            # block 2
            nn.Conv2d(16, 32, 3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            # output size = 64, width/4, length/4

            # # block 3
            nn.Conv2d(32, 64, 3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
            # # output size = 128, width/8 length/8
        )
        self.conv_size = int(
            64 * self.image_shape[0] / 8 * self.image_shape[1] / 8)
        self.theta_regression = nn.Sequential(
            # nn.Linear(self.conv_size, self.conv_size//2),
            # nn.ReLU(inplace=False),
            nn.Linear(self.conv_size, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 2 * 6 * self.num_slots),
            nn.Sigmoid()
        )

        # coordinate patching trick
        coord_map = net_util.create_coord_buffer(self.image_shape)
        self.register_buffer('coord_map_const', coord_map)

        constrain_add = torch.tensor(
            [[[self.min, 0., -1], [0., self.min, -1]]])
        self.register_buffer('constrain_add', constrain_add)
        constrain_mult = torch.tensor(
            [[[self.max - self.min, 0., 2.], [0., self.max - self.min, 2.]]])
        self.register_buffer('constrain_mult', constrain_mult)
        constrain_scale = torch.tensor([[[0.99, 0., 2.], [0., 0.99, 2.]]])
        self.register_buffer('constrain_scale', constrain_scale)
        constrain_shift = torch.tensor([[[0.01, 0., -1.], [0., 0.01, -1.]]])
        self.register_buffer('constrain_shift', constrain_shift)
        constrain_std = torch.tensor([[[1., 0., 1.], [0., 1., 1.]]])
        self.register_buffer('constrain_std', constrain_std)

    def forward(self, x):
        inp = torch.cat([x, self.coord_map_const.repeat(x.shape[0], 1, 1, 1)],
                        1)
        assert not torch.any(torch.isnan(inp)), 'theta 0'
        conv = self.detection_network(inp)
        assert not torch.any(torch.isnan(conv)), 'theta 1'
        conv = conv.view(-1, self.conv_size)
        theta = self.theta_regression(conv)
        assert not torch.any(torch.isnan(theta)), 'theta 2'
        theta = theta.view(-1, self.num_slots, 2, 2, 3)
        assert not torch.any(torch.isnan(theta)), 'theta 3'
        if self.constrain_theta:
            theta_mean = (theta[:, :, 0] * self.constrain_mult) + self.constrain_add
        else:
            theta_mean = (theta[:, :, 0] * self.constrain_scale) + self.constrain_shift
        theta_std = .1 * theta[:, :, 1] * self.constrain_std + 1e-4
        return theta_mean, \
               theta_std


class MaskNet(nn.Module):
    """
    Attention network for mask prediction
    """

    def __init__(self, in_channels=7, num_blocks=None, channel_base=32,
                 **kwargs):
        super().__init__()
        self.unet = net_util.UNet(num_blocks=num_blocks,
                                  in_channels=in_channels,
                                  out_channels=1,
                                  channel_base=channel_base)

    def forward(self, x):
        logits = self.unet(x)
        alpha = torch.sigmoid(logits)
        return alpha


class SpatialAutoEncoder(nn.Module):
    """
    Spatial transformation and reconstruction auto encoder
    """

    def __init__(self, latent_prior=1.0, latent_dim=8,
                 fg_sigma=0.11, bg_sigma=0.9, patch_shape=(32, 32),
                 image_shape=(256, 256),
                 num_blocks=2, **kwargs):
        super().__init__()
        self.prior = latent_prior
        self.fg_sigma = fg_sigma
        self.bg_sigma = bg_sigma
        self.patch_shape = patch_shape
        self.image_shape = image_shape

        self.encoding_network = EncoderNet(latent_dim//2, patch_shape)
        self.mask_encoding_network = EncoderNet(latent_dim//2, patch_shape)
        self.decoding_network = DecoderNet(latent_dim//2, patch_shape)
        self.mask_decoding_network = DecoderNet(latent_dim//2, patch_shape)
        self.mask_network = MaskNet(num_blocks=num_blocks)

        # self.spatial_network = SpatialLocalizationNet(conf)

    def forward(self, x, theta):
        mask, z, means, sigmas, kl_z, z_mask, mean_mask, sigma_mask, kl_mask = self.encode(x, theta)

        x_reconstruction, mask_pred = self.decode(z, z_mask, theta)

        mask_for_kl = net_util.invert(mask * 0.9998 + 0.0001, theta,
                                      self.image_shape)
        mask = net_util.invert(mask, theta, self.image_shape)

        # calculate mask prediction error
        # kl_mask_pred = net_util.kl_mask(
        #     mask_pred,
        #     mask_for_kl)
        scope = x[:, 6:, :, :]
        kl_mask_pred = net_util.reconstruction_likelihood(
                mask_for_kl.detach(),
                mask_pred,
                torch.ones_like(scope),
                self.fg_sigma,
                ).flatten(1)

        # calculate reconstruction error
        # mask = mask_pred * scope
        mask = mask * scope
        p_x = net_util.reconstruction_likelihood(
            x[:, :3],
            x_reconstruction,
            mask,
            self.fg_sigma)

        return x_reconstruction, \
               mask, \
               mask_pred, \
               torch.cat([z, z_mask], -1), \
               torch.cat([means, mean_mask], -1), \
               torch.cat([sigmas, sigma_mask], -1), \
               torch.cat([kl_z, kl_mask], -1), \
               p_x, \
               kl_mask_pred

    def encode(self, x, theta):
        # calculate spatial position for object from joint mask
        # and image
        grid = F.affine_grid(theta,
                             torch.Size((x.size()[0], 1, *self.patch_shape)))

        # get patch from theta and grid
        x_patch = net_util.transform(x, grid, theta)

        # generate object mask for patch
        mask = self.mask_network(x_patch)

        # concatenate x and new mask
        encoder_input = torch.cat([x_patch, mask.detach()], 1)

        # generate latent embedding of the patch
        mean_img, sigma_img = self.encoding_network(encoder_input)
        sigma_img = F.softplus(sigma_img) + 1e-10
        z_img, kl_z_img = net_util.differentiable_sampling(mean_img, sigma_img, self.prior)
        
        mean_mask, sigma_mask = self.mask_encoding_network(encoder_input)
        sigma_mask = F.softplus(sigma_mask) + 1e-10
        z_mask, kl_z_mask = net_util.differentiable_sampling(mean_mask, sigma_mask, self.prior)

        return mask, z_img, mean_img, sigma_img, kl_z_img, z_mask, mean_mask, sigma_mask, kl_z_mask

    def decode(self, z_img, z_mask, theta):
        decoded = self.decoding_network(z_img)
        mask_pred = self.mask_decoding_network(z_mask)[:, 3:]

        patch_reconstruction = torch.sigmoid(decoded[:, :3, :, :])

        # transform all components into original space
        x_reconstruction = net_util.invert(patch_reconstruction, theta,
                                           self.image_shape)
        mask_pred = torch.sigmoid(mask_pred) * 0.9998 + 0.0001
        mask_pred = net_util.invert(mask_pred, theta, self.image_shape)

        return x_reconstruction, mask_pred


class BackgroundEncoder(nn.Module):
    """
    Background encoder model which captures the rest of the picture
    TODO: Currently completely stupid
    """

    def __init__(self, image_shape=(256, 256), **kwargs):
        super().__init__()

        self.image_shape = image_shape

    def forward(self, x):
        loss = torch.zeros_like(x[:, 0, 0, 0])
        return torch.zeros_like(x[:, :3, :, :]), loss


class VAEBackgroundModel(nn.Module):
    """
    Background encoder model which captures the rest of the picture
    Not completely stupid anymore
    """

    def __init__(self, image_shape=(256, 256)):
        super().__init__()
        self.image_shape = image_shape
        self.enc_net = EncoderNet(3, image_shape, 4)
        self.dec_net = DecoderNet(32, image_shape)
        self.prior = 1.

    def forward(self, x, scope):
        # concatenate x and new mask
        encoder_input = torch.cat([x, scope], 1)

        # generate latent embedding of the patch
        mean, sigma = self.enc_net(encoder_input)
        z, kl_z = net_util.differentiable_sampling(mean, sigma, self.prior)
        decoded = self.dec_net(z)
        return decoded[:, :, :, :], kl_z

    def init_bias(self, images):
        with torch.no_grad():
            self.dec_net.network[-1].bias.zero_()
            norm = len(images)
            append = torch.zeros((1, self.image_shape[0], self.image_shape[1]))
            for image in images:
                # for image in image_batch[0]:
                fill = torch.cat([image, append], 0).cuda()
                self.dec_net.network[-1].bias += torch.mean(
                    fill.view(-1, 1))
            self.dec_net.network[-1].bias /= norm


class FCBackgroundModel(nn.Module):
    """
    Background encoder model working on a single fully connected layer
    """

    def __init__(self, image_shape=(256, 256), **kwargs):
        super().__init__()

        self.image_shape = image_shape
        image_shape_flattened = self.image_shape[0] * self.image_shape[1] * 4
        self.net = nn.Linear(1, image_shape_flattened, bias=False)

    def init_bias(self, images):
        with torch.no_grad():
            self.net.weight.zero_()
            norm = 0.
            append = torch.zeros((1, self.image_shape[0], self.image_shape[1]))
            for image in images:
                fill = torch.cat([image, append], 0).cuda()
                self.net.weight.data += fill.view(-1, 1)
                norm += 1.
            self.net.weight.data /= norm

    def forward(self, x, scope):
        encoder_shaping = torch.cat([x, scope], 1)
        _input = torch.ones((x.size()[0], 1)).cuda()
        output = self.net(_input)
        dummy_loss = torch.tensor([0.0] * x.size()[0]).cuda()
        return output.view(encoder_shaping.size()), dummy_loss


class MaskedAIR(nn.Module):
    """
    Full model for image reconstruction and mask prediction
    """

    def __init__(
            self, bg_sigma=0.01, fg_sigma=0.05, latent_prior=1.,
            latent_dim=16, patch_shape=(16, 16),
            image_shape=(256, 256), num_blocks=2, num_slots=8,
            constrain_theta=False, beta=1., gamma=1., 
            predict_masks=False, **kwargs):
        super().__init__()
        self.bg_sigma = bg_sigma
        self.fg_sigma = fg_sigma
        self.latent_prior = latent_prior
        self.latent_dim = latent_dim
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.constrain_theta = constrain_theta
        self.predict_masks = predict_masks
        self.spatial_vae = SpatialAutoEncoder(latent_prior,
                                              latent_dim,
                                              fg_sigma,
                                              bg_sigma,
                                              patch_shape,
                                              image_shape,
                                              num_blocks)
        # self.background_model = BackgroundEncoder(conf)
        # self.background_model = VAEBackgroundModel(conf)
        self.background_model = FCBackgroundModel(image_shape)
        # self.mask_background = MaskNet(conf, 6, 4)
        self.spatial_localization_net = SpatialLocalizationNet(image_shape,
                                                               patch_shape,
                                                               constrain_theta,
                                                               num_slots)

        self.beta = beta
        self.gamma = gamma

        self.running = 0

        self.graph_depth = latent_dim + 8
        self.graph_size = (num_slots, num_slots)

    def init_background_weights(self, images):
        self.background_model.init_bias(images)

    def build_image_graph(self, x):
        """
        Builds the graph representation of an image from one forward pass
        of the model
        """
        embeddings, loss = self.build_flat_image_representation(x)

        embeddings_interaction = embeddings.unsqueeze(2)
        grid_interactions = embeddings_interaction - \
                            embeddings_interaction.permute(0, 2, 1, 3)

        # grid_embeddings = grid_interactions + embedding_matrix

        return grid_interactions, embeddings, loss

    def forward(self, x):
        """
        Main model forward pass
        """
        # initialize arrays for visualization
        masks = []
        mask_preds = []
        latents = []
        latents_mean = []
        latents_std = []

        # initialize loss components
        scope = torch.ones_like(x[:, :1])
        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        kl_zs = torch.zeros_like(
            x[:, 0, :self.latent_dim, 0]).squeeze()
        kl_masks = torch.zeros_like(x[:, 0, :, :]).view(x.shape[0],
                                                        x.shape[2] * x.shape[
                                                            3])
        p_x_loss = torch.zeros_like(x)

        background, _ = self.background_model(x, scope)
        background = background[:, :3, :, :]

        # print(self.spatial_localization_net.detection_network[0].bias.mean())

        # get all thetas at once
        inp = torch.cat([x, (x - background).detach()], 1)

        thetas_mean, thetas_std = self.spatial_localization_net(inp)
        thetas = []
        # construct the patchwise shaping of the model
        for i in range(self.num_slots):
            theta = dists.Normal(thetas_mean[:, i], thetas_std[:, i]).rsample()
            theta = thetas_mean[:, i]
            thetas.append(theta)
            inp = torch.cat(
                [x, ((x - background) - total_reconstruction).detach(), scope],
                1)
            x_recon, mask, mask_pred, z, means, sigmas, kl_z, p_x, kl_m = self.spatial_vae(
                inp, theta)
            scope = scope - mask
            kl_zs += kl_z
            p_x_loss += p_x
            kl_masks += kl_m
            total_reconstruction += mask * x_recon

            # save for visualization
            masks.append(mask)
            mask_preds.append(mask_pred)
            latents.append(z)
            latents_mean.append(means)
            latents_std.append(sigmas)
            assert torch.all(torch.isfinite(total_reconstruction))

        total_reconstruction += background * scope
        assert torch.all(torch.isfinite(total_reconstruction))
        # calculate reconstruction error
        p_x = net_util.reconstruction_likelihood(x,
                                                 background,
                                                 (1 - torch.sum(
                                                     torch.cat(masks, 1), 1,
                                                     True)),
                                                 self.bg_sigma,
                                                 self.running)
        p_x_loss += p_x

        thetas = torch.cat(thetas, 1)

        # calculate the final loss
        loss = -1 * p_x_loss.sum([1, 2, 3]) + \
               self.beta * kl_zs.sum([1]) + \
               -1 * self.gamma * kl_masks.sum([1])
        assert not torch.any(torch.isnan(thetas)), 'thetas nan'
        assert not torch.any(torch.isnan(p_x_loss)), 'p_x_loss nan'
        assert not torch.any(torch.isnan(kl_zs)), 'kl_zs nan'
        assert not torch.any(torch.isnan(kl_masks)), 'kl_masks nan'

        # torchify all outputs
        masks.insert(0, scope)
        masks = torch.cat(masks, 1)
        mask_pred_scope = torch.ones_like(scope)
        for mask in mask_preds:
            mask_pred_scope = mask_pred_scope * (1 - mask)
        mask_preds.insert(0, mask_pred_scope)
        mask_preds = torch.cat(mask_preds, 1)
        latents = torch.stack(latents, 1)
        latents_mean = torch.stack(latents_mean, 1)
        latents_std = torch.stack(latents_std, 1)

        self.running += 1

        # currently missing is the mask reconstruction loss
        return_dict = {'loss': loss,
                       'reconstructions': total_reconstruction,
                       'p_x_loss': p_x_loss,
                       'masks': masks,
                       'mask_preds': mask_preds,
                       'latents': latents,
                       'latents_mean': latents_mean,
                       'latents_std': latents_std,
                       'theta': thetas,
                       'thetas_mean': thetas_mean,
                       'thetas_std': thetas_std,
                       'mask_loss': kl_masks,
                       'kl_loss': kl_zs}
        return loss, return_dict
    
    def build_flat_image_representation(self, x, return_dists=False):
        loss, res = self.forward(x)
        masks = res['masks']
        latents = res['latents']
        latents_mean = res['latents_mean']
        latents_std = res['latents_std']
        pos = res['theta']
        pos_mean = res['thetas_mean']
        pos_std = res['thetas_std']
        
        # offset necessary for dark masks
        grid = (net_util.center_of_mass(masks[:, 1:] + 1e-20) - (self.image_shape[0]/2)) / self.image_shape[0]
        assert not torch.any(torch.isnan(grid))

        if return_dists:
            full_mean = torch.cat(
                [grid, pos_mean.view(-1, self.num_slots, 6), latents_mean], -1)
            flatten_pos_std = pos_std.view(-1, self.num_slots, 6)
            grid_std_x = flatten_pos_std[..., 4:5]
            grid_std_y = flatten_pos_std[..., 5:6]
            full_std = torch.cat(
                [grid_std_x, grid_std_y, flatten_pos_std, latents_std], -1)
            assert torch.all(full_std > 0)
            # print(torch.mean(loss))
            return full_mean, full_std
        else:
            full = torch.cat(
                [grid, pos.view(-1, self.num_slots, 6), latents_mean, latents_std], -1)
            return full, loss

    def reconstruct_from_latent(self, x, imgs=None, reconstruct_mask=True):
        """
        Given a latent representation of the image, construct a full image again

        Inputs:
            - x: torch.Tensor shapes (batch, num_slots, latent_dims + 6[theta]
            + 2[pos])
        """
        reconstruct_masks = reconstruct_mask or self.predict_masks
        if not reconstruct_mask and imgs is None:
            raise ValueError('Cannot compute masks without image')
        images = torch.zeros(x.shape[0], 3, self.image_shape[0], self.image_shape[1]).cuda()
        if imgs is not None:
            images = torch.zeros_like(imgs)
            p_x = torch.zeros_like(images)
        scope = torch.ones_like(images[:, 0:1, :, :])
        if not reconstruct_mask:
            kl_mask = torch.zeros_like(scope)

        latents = x[:, :, 8:self.latent_dim + 8]
        thetas = x[:, :, 2:8]
        # print(x.shape)
        # print(thetas.shape)
        thetas = thetas.contiguous().view(-1, self.num_slots, 2, 3)
        masks = []

        background, _ = self.background_model(images, scope)
        background = background[:, :3, :, :]

        for i in range(self.num_slots):
            recon, mask = self.spatial_vae.decode(
                    latents[:, i, :self.latent_dim//2], 
                    latents[:, i, self.latent_dim//2:], 
                    thetas[:, i, :])
            if not reconstruct_mask:
                mask_pred = mask
                inp = torch.cat([imgs, ((imgs - background) - images).detach(), scope], 1)
                grid = F.affine_grid(thetas[:, i], torch.Size((imgs.size()[0], 1, *self.patch_shape)))
                x_patch = net_util.transform(inp, grid, thetas[:, i])
                mask = self.spatial_vae.mask_network(x_patch)
                mask = net_util.invert(mask, thetas[:, i], self.image_shape)

                # calculate mask prediction error
                # kl_mask = net_util.kl_mask(
                #     mask_pred,
                #     mask_for_kl)
                kl_mask_pred = net_util.reconstruction_likelihood(
                        mask.detach(),
                        mask_pred,
                        torch.ones_like(scope),
                        self.fg_sigma,
                        )
                kl_mask += kl_mask_pred
            images += recon * mask * scope
            if imgs is not None:
                p_x += net_util.reconstruction_likelihood(
                    imgs, 
                    recon, 
                    mask * scope,
                    self.fg_sigma, 
                    self.running)
            masks.append(mask * scope)
            scope = scope * (1 - mask)

        images += scope * background

        if imgs is not None:
            p_x += net_util.reconstruction_likelihood(
                imgs, 
                background,
                (1 - torch.sum(torch.cat(masks, 1), 1, True)),
                self.bg_sigma, self.running)
            if not reconstruct_mask:
                return images, p_x, kl_mask
            return images, p_x, torch.zeros_like(scope)

        return images
