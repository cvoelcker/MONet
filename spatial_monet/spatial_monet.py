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

    def __init__(self, component_latent_dim=32, patch_shape=(32, 32),
                 input_size=8,
                 **kwargs):
        super().__init__()

        self.latent_dim = component_latent_dim
        self.patch_shape = patch_shape

        self.network = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # nn.Conv2d(64, 64, 3, padding=(1,1)),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
        )
        self.conv_size = int(
            64 * self.patch_shape[0] / (2 ** 3) * self.patch_shape[1] / (
                    2 ** 3))
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_size, 2 * self.latent_dim),
            nn.ReLU(inplace=True))
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
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

    def __init__(self, component_latent_dim=32, patch_shape=(32, 32),
                 **kwargs):
        super().__init__()

        self.latent_dim = component_latent_dim
        self.patch_shape = patch_shape

        # gave it inverted hourglass shape
        # maybe that helps (random try)
        self.network = nn.Sequential(
            nn.Conv2d(self.latent_dim + 2, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=(2, 2)),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 5, padding=(2,2)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 32, 3, padding=(1,1)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # output size = 32, width/2, length/2

            # block 2
            nn.Conv2d(16, 32, 3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # output size = 64, width/4, length/4

            # # block 3
            nn.Conv2d(32, 64, 3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # # output size = 128, width/8 length/8
        )
        self.conv_size = int(
            64 * self.image_shape[0] / 8 * self.image_shape[1] / 8)
        self.theta_regression = nn.Sequential(
            # nn.Linear(self.conv_size, self.conv_size//2),
            # nn.ReLU(inplace=True),
            nn.Linear(self.conv_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6 * self.num_slots),
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

    def theta_restrict(self, theta):
        return F.mse_loss(theta * self.theta_mask, self.theta_mean)

    def forward(self, x):
        inp = torch.cat([x, self.coord_map_const.repeat(x.shape[0], 1, 1, 1)],
                        1)
        conv = self.detection_network(inp)
        conv = conv.view(-1, self.conv_size)
        theta = self.theta_regression(conv)
        theta = theta.view(-1, self.num_slots, 2, 3)
        if self.constrain_theta:
            theta = (theta * self.constrain_mult) + self.constrain_add
        else:
            theta = (theta * self.constrain_scale) + self.constrain_shift
        return theta


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

    def __init__(self, latent_prior=1.0, component_latent_dim=8,
                 fg_sigma=0.11, bg_sigma=0.9, patch_shape=(32, 32),
                 image_shape=(256, 256),
                 num_blocks=2, **kwargs):
        super().__init__()
        self.prior = latent_prior
        self.fg_sigma = fg_sigma
        self.bg_sigma = bg_sigma
        self.patch_shape = patch_shape
        self.image_shape = image_shape

        self.encoding_network = EncoderNet(component_latent_dim, patch_shape)
        self.decoding_network = DecoderNet(component_latent_dim, patch_shape)
        self.mask_network = MaskNet(num_blocks=num_blocks)

        # self.spatial_network = SpatialLocalizationNet(conf)

    def forward(self, x, theta):
        z, kl_z, mask = self.encode(x, theta)

        x_reconstruction, mask_pred = self.decode(z, theta)

        mask_for_kl = net_util.invert(mask * 0.9998 + 0.0001, theta,
                                      self.image_shape)
        mask = net_util.invert(mask, theta, self.image_shape)

        # calculate mask prediction error
        kl_mask_pred = net_util.kl_mask(
            mask_pred,
            mask_for_kl)

        # calculate reconstruction error
        scope = x[:, 6:, :, :]
        mask = mask * scope
        p_x = net_util.reconstruction_likelihood(
            x[:, :3],
            x_reconstruction,
            mask,
            self.fg_sigma)

        return x_reconstruction, mask, z, kl_z, p_x, kl_mask_pred

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
        mean, sigma = self.encoding_network(encoder_input)
        z, kl_z = net_util.differentiable_sampling(mean, sigma, self.prior)

        return z, kl_z, mask

    def decode(self, z, theta):
        decoded = self.decoding_network(z)

        patch_reconstruction = decoded[:, :3, :, :]
        mask_pred = decoded[:, 3:, :, :]

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
            norm = len(images.dataset)
            append = torch.zeros((1, self.image_shape[0], self.image_shape[1]))
            for image_batch in images:
                for image in image_batch[0]:
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
            norm = len(images.dataset)
            append = torch.zeros((1, self.image_shape[0], self.image_shape[1]))
            for image_batch in images:
                for image in image_batch:
                    fill = torch.cat([image, append], 0).cuda()
                    self.net.weight.data += fill.view(-1, 1)
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
            component_latent_dim=16, patch_shape=(16, 16),
            image_shape=(256, 256), num_blocks=2, num_slots=8,
            constrain_theta=False, beta=1., gamma=1., **kwargs):
        super().__init__()
        self.bg_sigma = bg_sigma
        self.fg_sigma = fg_sigma
        self.latent_prior = latent_prior
        self.component_latent_dim = component_latent_dim
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.constrain_theta = constrain_theta
        self.spatial_vae = SpatialAutoEncoder(latent_prior,
                                              component_latent_dim,
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

        self.graph_depth = component_latent_dim + 8
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

    def build_flat_image_representation(self, x):
        loss, _, _, masks, embeddings, positions, _, _ = self.forward(
            x).values()
        grid = net_util.center_of_mass(masks[:, 1:])
        full = torch.cat(
            [embeddings, positions.view(-1, self.num_slots, 6), grid], -1)
        return full, loss

    def forward(self, x):
        """
        Main model forward pass
        """
        # initialize arrays for visualization
        masks = []
        latents = []

        # initialize loss components
        scope = torch.ones_like(x[:, :1])
        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        kl_zs = torch.zeros_like(
            x[:, 0, :self.component_latent_dim, 0]).squeeze()
        kl_masks = torch.zeros_like(x[:, 0, :, :]).view(x.shape[0],
                                                        x.shape[2] * x.shape[
                                                            3])
        p_x_loss = torch.zeros_like(x)

        background, _ = self.background_model(x, scope)
        background = background[:, :3, :, :]

        # get all thetas at once
        inp = torch.cat([x, (x - background).detach()], 1)

        thetas = self.spatial_localization_net(inp)

        # construct the patchwise shaping of the model
        for i in range(self.num_slots):
            theta = thetas[:, i]
            inp = torch.cat(
                [x, ((x - background) - total_reconstruction).detach(), scope],
                1)
            x_recon, mask, z, kl_z, p_x, kl_m = self.spatial_vae(inp, theta)
            scope = scope - mask
            kl_zs += kl_z
            p_x_loss += p_x
            kl_masks += kl_m
            total_reconstruction += mask * x_recon

            # save for visualization
            masks.append(mask)
            latents.append(z)

        total_reconstruction += background * scope
        # calculate reconstruction error
        p_x = net_util.reconstruction_likelihood(x,
                                                 background,
                                                 (1 - torch.sum(
                                                     torch.cat(masks, 1), 1,
                                                     True)),
                                                 self.bg_sigma,
                                                 self.running)
        p_x_loss += p_x

        # calculate the final loss
        loss = -p_x_loss.sum(
            [1, 2, 3]) + self.beta * kl_zs.sum(
            [1]) + self.gamma * kl_masks.sum([1])

        # torchify all outputs
        masks.insert(0, scope)
        masks = torch.cat(masks, 1)
        latents = torch.stack(latents, 1)

        self.running += 1

        # currently missing is the mask reconstruction loss
        return_dict = {'loss': loss,
                       'reconstructions': total_reconstruction,
                       'reconstruction_loss': p_x_loss,
                       'masks': masks,
                       'latents': latents,
                       'theta': thetas,
                       'mask_loss': kl_masks,
                       'kl_loss': kl_zs}
        return return_dict

    def reconstruct_from_latent(self, x):
        '''
        Given a latent representation of the image, construct a full image again

        Inputs:
            - x: torch.Tensor shapes (batch, num_slots, latent_dims + 6[theta] + 2[pos])
        '''
        images = self.zeros(x.shape[0], 3, self.image_shape[1], self.image_shape[2])
        scope = self.ones(x.shape[0], 1, self.image_shape[1], self.image_shape[2])

        loss, _, _, masks, embeddings, positions, _, _ = self.forward(
            x).values()
        grid = net_util.center_of_mass(masks[:, 1:])
        full = torch.cat(
            [embeddings, grid, positions.view(-1, self.num_slots, 6)], -1)
        latents = x[:, :, :self.component_latent_dim]
        thetas = x[:, :, self.component_latent_dim+2:]
        thetas = thetas.view(-1, self.num_slots, 2, 3)

        background = self.background_model(images)

        for i in range(self.num_slots):
            recon, mask = self.spatial_vae.decode(x[:, i, :], thetas[:, i, :])
            images += recon * mask * recon
            scope = scope * (1 - mask)
        
        images += scope * background

        return images
