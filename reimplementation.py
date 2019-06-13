import collections
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from model import UNet


def create_coord_buffer(patch_shape):
    ys = torch.linspace(-1, 1, patch_shape[0])
    xs = torch.linspace(-1, 1, patch_shape[1])
    xv, yv = torch.meshgrid(ys, xs)
    coord_map = torch.stack((xv, yv)).unsqueeze(0)
    return coord_map


def alternate_inverse(theta):
    inv_theta = torch.zeros_like(theta)
    inv_theta[:, 0, 0] = 1/theta[:, 0, 0]
    inv_theta[:, 1, 1] = 1/theta[:, 1, 1]
    inv_theta[:, 0, 2] = -theta[:, 0, 2]/theta[:, 0, 0]
    inv_theta[:, 1, 2] = -theta[:, 1, 2]/theta[:, 1, 1]
    return inv_theta


def differentiable_sampling(mean, sigma, prior_sigma):
    dist = dists.Normal(mean, sigma)
    dist_0 = dists.Normal(0., sigma)
    z = mean + dist_0.sample()
    kl_z = dists.kl_divergence(dist, dists.Normal(0., prior_sigma))
    kl_z = torch.sum(kl_z, 1)
    return z, kl_z


def reconstruction_likelihood(x, recon, mask, sigma):
    dist = dists.Normal(x, sigma)
    p_x = dist.log_prob(recon)
    p_x = torch.sum(p_x * mask, [1, 2, 3])
    return p_x


def kl_mask(mask, mask_pred):
    tr_masks = torch.transpose(mask, 1, 3)
    tr_masks = (torch.transpose(tr_masks, 1, 2) + 0.0001).squeeze()
    tr_mask_preds = torch.transpose(mask_pred, 1, 3)
    tr_mask_preds = torch.transpose(tr_mask_preds, 1, 2).squeeze()

    q_masks = dists.Categorical(probs=tr_masks)
    q_masks_recon = dists.Categorical(logits=tr_mask_preds)
    kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
    kl_masks = torch.sum(kl_masks, [1])
    if torch.any(torch.isnan(kl_masks)):
        pickle.dump(tr_masks.cpu().data.numpy(), open('save_tr_mask_failed', 'wb'))
        pickle.dump(tr_mask_preds.cpu().data.numpy(), open('save_tr_mask_preds_failed', 'wb'))
        raise ValueError
    return kl_masks
    

def transform(x, grid, theta):
    x = F.grid_sample(x, grid)
    return x


def invert(x, theta, image_shape):
    # theta_expanded = torch.cat([theta, self.theta_append.repeat(x.size()[0], 1, 1)], dim=1)
    # inverse_theta = theta_expanded.inverse()
    inverse_theta = alternate_inverse(theta)
    if x.size()[1] == 1:
        size = torch.Size((x.size()[0], 1, *image_shape))
    elif x.size()[1] == 3:
        size = torch.Size((x.size()[0], 3, *image_shape))
    grid = F.affine_grid(inverse_theta, size)
    x = F.grid_sample(x, grid)
    return x


class EncoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """
    def __init__(self, conf):
        super().__init__()

        self.latent_dim = conf.component_latent_dim
        self.patch_shape = conf.patch_shape

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 64, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            )
        self.conv_size = int(64 * self.patch_shape[0]/(2**4) * self.patch_shape[1]/(2**4))
        self.mlp = nn.Sequential(
                nn.Linear(self.conv_size, 2 * self.latent_dim),
                nn.ReLU(inplace=True))
        self.mean_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid())

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
    def __init__(self, conf):
        super().__init__()
        
        self.latent_dim = conf.component_latent_dim
        self.patch_shape = conf.patch_shape
        
        # gave it inverted hourglass shape
        # maybe that helps (random try)
        self.network = nn.Sequential(
            nn.Conv2d(self.latent_dim + 2, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )

        # coordinate patching trick
        coord_map = create_coord_buffer(self.patch_shape)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, x):
        # adds coordinate information to z and 
        # produces a tiled representation of z
        z_scaled = x.unsqueeze(-1).unsqueeze(-1)
        z_tiled = z_scaled.repeat(1, 1, self.patch_shape[0], self.patch_shape[1])
        coord_map = self.coord_map_const.repeat(x.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.network(inp)
        return result


class SpatialLocalizationNet(nn.Module):
    """
    Attention network for object localization
    """
    def __init__(self, conf):
        super().__init__()

        self.image_shape = conf.image_shape
        self.patch_shape = conf.patch_shape
        self.batch_size = conf.batch_size
        self.constrain_theta = conf.constrain_theta
        self.num_slots = conf.num_slots
        
        self.detection_network = nn.Sequential(
                # block 1
                nn.Conv2d(9, 16, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 32, width/2, length/2

                # block 2
                nn.Conv2d(16, 32, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 64, width/4, length/4

                # # block 3
                # nn.Conv2d(23, 32, 3, padding=(1,1)),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(2, stride=2),
                # # output size = 128, width/8 length/8
                )
        self.conv_size = int(32 * self.image_shape[0]/4 * self.image_shape[1]/4)
        self.theta_regression = nn.Sequential(
                nn.Linear(self.conv_size, 128, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(128, 6 * self.num_slots, bias=False),
                nn.Sigmoid()
                )
        
        # coordinate patching trick
        coord_map = create_coord_buffer(self.image_shape).repeat(self.batch_size,1,1,1)
        self.register_buffer('coord_map_const', coord_map)
        
        constrain_mask = torch.tensor([[[0.5, 0., -0.5], [0., 0.5, -0.5]]])#.repeat(self.batch_size, 1, 1)
        self.register_buffer('constrain_mask', constrain_mask)
        constrain_stretch = torch.tensor([[[0.1, 0., 2.], [0., 0.1, 2.]]])#.repeat(self.batch_size, 1, 1)
        self.register_buffer('constrain_stretch', constrain_stretch)

    def theta_restrict(self, theta):
        return F.mse_loss(theta * self.theta_mask, self.theta_mean)

    def forward(self, x):
        inp = torch.cat([x, self.coord_map_const], 1)
        conv = self.detection_network(inp)
        conv = conv.view(-1, self.conv_size)
        theta = self.theta_regression(conv)
        theta = theta.view(-1, self.num_slots, 2, 3)
        if self.constrain_theta:
            theta = ((theta) + self.constrain_mask) * self.constrain_stretch
        return theta



class MaskNet(nn.Module):
    """
    Attention network for mask prediction
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet(num_blocks=conf.num_blocks,
                         in_channels=4,
                         out_channels=2,
                         channel_base=conf.channel_base)

    def forward(self, x):
        logits = self.unet(x)
        alpha = torch.softmax(logits, 1)
        return alpha


class SpatialAutoEncoder(nn.Module):
    """
    Spatial transformation and reconstruction auto encoder
    """
    def __init__(self, conf):
        super().__init__()
        self.prior = conf.latent_prior
        self.fg_sigma = conf.fg_sigma
        self.patch_shape = conf.patch_shape
        self.image_shape = conf.image_shape

        self.encoding_network = EncoderNet(conf)
        self.decoding_network = DecoderNet(conf)
        self.mask_network = MaskNet(conf)
        # self.spatial_network = SpatialLocalizationNet(conf)
        
        self.x_reconstruction = None
        self.mask_transformation = None
        self.mask_prediction = None
        self.scope = None
        self.z = None
        self.kl_z = None
        self.theta_loss = None
        self.p_x = None
        self.kl_mask = None

    def forward(self, x, theta):
        # calculate spatial position for object from joint mask
        # and image
        grid = F.affine_grid(theta, torch.Size((x.size()[0], 4, *self.patch_shape)))

        # get patch from theta and grid
        x_patch = transform(x[:, :4, :, :], grid, theta)

        # generate object mask for patch
        # TODO: this seems highly questionable, check again
        alpha = self.mask_network(x_patch)
        old_scope = x_patch[:, 3:, :, :]
        x_patch = x_patch[:, :3, :, :]
        mask = old_scope * alpha[:, :1]
        new_scope = old_scope * alpha[:, 1:]

        # concatenate x and new mask
        encoder_input = torch.cat([x_patch, mask], 1)

        # generate latent embedding of the patch
        mean, sigma = self.encoding_network(encoder_input)
        z, kl_z = differentiable_sampling(mean, sigma, self.prior)

        decoded = self.decoding_network(z)
        
        # seperate patch into img and scope
        patch_reconstruction = decoded[:, :3, :, :] * mask
        mask_pred = decoded[:, 3:, :, :]

        # transform all components into original space
        x_reconstruction = invert(patch_reconstruction, theta, self.image_shape)
        mask_transformation = invert(mask, theta, self.image_shape)
        mask_prediction = invert(mask_pred, theta, self.image_shape)
        
        scope = x[:, 3:4, :, :]
        scope = scope - mask_transformation

        # calculate reconstruction error
        p_x = reconstruction_likelihood(x[:, :3], x_reconstruction, 
                mask_transformation, self.fg_sigma)

        # calculate mask prediction error
        kl_mask_pred = kl_mask(mask_transformation, mask_prediction)
        
        self.x_reconstruction = x_reconstruction
        self.mask_transformation = mask_transformation
        self.mask_prediction = mask_prediction
        self.scope = scope
        self.z = z
        self.kl_z = kl_z
        self.p_x = p_x
        self.kl_mask = kl_mask_pred
        return x_reconstruction


class BackgroundEncoder(nn.Module):
    """
    Background encoder model which captures the rest of the picture
    TODO: Currently completely stupid
    """
    def __init__(self, conf):
        super().__init__()

        self.image_shape = conf.image_shape

    def forward(self, x):
        loss = torch.zeros_like(x[:, 0, 0, 0])
        return torch.zeros_like(x[:, :3, :, :]), loss


class VAEBackgroundModel(nn.Module):
    """
    Background encoder model which captures the rest of the picture
    Not completely stupid anymore
    """
    def __init__(self, conf):
        super().__init__()
        fg_components = conf.component_latent_dim
        patch_shape = conf.patch_shape
        conf.component_latent_dim = 1
        conf.patch_shape = (128, 128)
        self.enc_net = EncoderNet(conf)
        self.dec_net = DecoderNet(conf)
        conf.component_latent_dim = fg_components
        conf.patch_shape = patch_shape
        self.image_shape = conf.image_shape
        self.prior = 0.01

    def forward(self, x, scope):
        # concatenate x and new mask
        encoder_input = torch.cat([x, scope], 1)

        # generate latent embedding of the patch
        mean, sigma = self.enc_net(encoder_input)
        z, kl_z = differentiable_sampling(mean, sigma, self.prior)
        decoded = self.dec_net(z)
        return decoded[:, :, :, :], kl_z


class MaskedAIR(nn.Module):
    """
    Full model for image reconstruction and mask prediction
    """
    def __init__(self, conf):
        super().__init__()

        self.bg_sigma = conf.bg_sigma

        self.spatial_vae = SpatialAutoEncoder(conf)
        # self.background_model = BackgroundEncoder(conf)
        self.background_model = VAEBackgroundModel(conf)
        self.mask_background = MaskNet(conf)
        self.spatial_localization_net = SpatialLocalizationNet(conf)

        self.num_slots = conf.num_slots

        self.beta = conf.beta
        self.gamma = conf.gamma
        self.kappa = 0.5
    
    def forward(self, x):
        # initialize arrays for visualization
        masks = []
        latents = []
        mask_preds = []

        # initialize loss components
        scope = torch.ones_like(x[:, 0:1])
        all_scopes = []
        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        kl_zs = torch.zeros_like(loss)
        # theta_losses = torch.zeros_like(loss)
        kl_masks = torch.zeros_like(loss)
        p_x_loss = torch.zeros_like(loss)
        
        # compute background at the beginning of the damn thing
        background_mask, kl_z = self.background_model(x, scope)
        background = background_mask[:, :3, :, :]
        mask_channel = background_mask[:, 3:, :, :]
        mask_input = torch.cat([x, scope], 1)
        alpha = self.mask_background(mask_input)
        mask = scope * alpha[:, :1]
        scope = scope * alpha[:, 1:]
        all_scopes.append(scope)
        masks.append(mask)
        mask_preds.append(mask_channel)
        kl_zs += kl_z
        total_reconstruction += background * mask
        
        # calculate reconstruction error
        p_x_loss += reconstruction_likelihood(x, background, 
                mask, self.bg_sigma)

        # calculate mask prediction error
        kl_masks += kl_mask(mask, mask_channel)

        # get all thetas at once
        inp = torch.cat([x, x - total_reconstruction, scope], 1)
        thetas = self.spatial_localization_net(inp)

        # construct the patchwise shaping of the model
        for i in range(self.num_slots):
            theta = thetas[:, i]
            scope = all_scopes[-1]
            inp = torch.cat([x, scope, x-total_reconstruction], 1)
            x_recon = self.spatial_vae(inp, theta)
            mask = self.spatial_vae.mask_transformation
            mask_pred = self.spatial_vae.mask_prediction
            all_scopes.append(self.spatial_vae.scope)
            z = self.spatial_vae.z
            kl_zs += self.spatial_vae.kl_z
            p_x_loss += self.spatial_vae.p_x
            kl_masks += self.spatial_vae.kl_mask
            # theta_losses = theta_losses + self.spatial_vae.theta_loss
            total_reconstruction = total_reconstruction + mask * x_recon
            
            # save for visualization
            masks.append(mask)
            mask_preds.append(mask_pred)
            latents.append(z)
       
        # torchify all lists
        scope = all_scopes[-1]
        masks.append(scope)
        masks = torch.cat(masks, 1)
        latents = torch.cat(latents, 1)

        # # compute background
        # inp_background = torch.cat([x, scope], 1)
        # background, kl_z = self.background_model(x, scope)
        # kl_zs += kl_z
        # total_reconstruction += background[:, :3, :, :] * scope

        # calculate reconstruction error
        p_x_loss += reconstruction_likelihood(x, total_reconstruction, 
                torch.ones_like(scope), self.bg_sigma)

        # get mask entropy loss
        entropy = F.softmax(masks, 1) * F.log_softmax(masks, 1)
        entropy = -100.0 * torch.mean(torch.sum(entropy, 1), [1,2])
        
        loss += - p_x_loss + entropy + self.beta * kl_zs# + self.kappa * kl_masks
        
        # currently missing is the mask reconstruction loss
        return {'loss': loss,
                'reconstructions': total_reconstruction,
                'reconstruction_loss': p_x_loss,
                'masks': masks,
                'latents': latents,}
