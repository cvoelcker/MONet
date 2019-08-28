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
    dist_0 = dists.Normal(0., prior_sigma)
    z = mean + sigma * dist_0.sample()
    kl_z = dists.kl_divergence(dist, dist_0)
    kl_z = torch.mean(kl_z, 1)
    return z, kl_z


def reconstruction_likelihood(x, recon, mask, sigma, i=None):
    dist = dists.Normal(x, sigma)
    p_x = dist.log_prob(recon) * mask
    return p_x


def kl_mask(mask, mask_pred):
    tr_masks = mask.view(mask.size()[0], -1)
    tr_mask_preds = mask_pred.view(mask_pred.size()[0], -1)
    
    q_masks = dists.Bernoulli(probs=tr_masks)
    q_masks_recon = dists.Bernoulli(logits=tr_mask_preds)
    kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
    kl_masks = torch.mean(kl_masks, [1])
    return kl_masks
    

def transform(x, grid, theta):
    x = F.grid_sample(x, grid)
    return x


def invert(x, theta, image_shape, padding='zeros'):
    inverse_theta = alternate_inverse(theta)
    if x.size()[1] == 1:
        size = torch.Size((x.size()[0], 1, *image_shape))
    elif x.size()[1] == 3:
        size = torch.Size((x.size()[0], 3, *image_shape))
    grid = F.affine_grid(inverse_theta, size)
    x = F.grid_sample(x, grid, padding_mode=padding)

    return x


class EncoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """
    def __init__(self, conf, input_size=8):
        super().__init__()

        self.latent_dim = conf.component_latent_dim
        self.patch_shape = conf.patch_shape

        self.network = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # nn.Conv2d(64, 64, 3, padding=(1,1)),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
            )
        self.conv_size = int(64 * self.patch_shape[0]/(2**3) * self.patch_shape[1]/(2**3))
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
    def __init__(self, conf):
        super().__init__()
        
        self.latent_dim = conf.component_latent_dim
        self.patch_shape = conf.patch_shape
        
        # gave it inverted hourglass shape
        # maybe that helps (random try)
        self.network = nn.Sequential(
            nn.Conv2d(self.latent_dim + 2, 32, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=(2,2)),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 5, padding=(2,2)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(1,1)),
            nn.ReLU(inplace=True),
            #nn.Conv2d(64, 32, 3, padding=(1,1)),
            #nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
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

        self.min = 0.05
        self.max = 0.2
        
        self.detection_network = nn.Sequential(
                # block 1
                nn.Conv2d(8, 16, 3, stride=1, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 32, width/2, length/2

                # block 2
                nn.Conv2d(16, 32, 3, stride=1, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 64, width/4, length/4

                # # block 3
                nn.Conv2d(32, 64, 3, stride=1, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # # output size = 128, width/8 length/8
                )
        self.conv_size = int(64 * self.image_shape[0]/8 * self.image_shape[1]/8)
        self.theta_regression = nn.Sequential(
                # nn.Linear(self.conv_size, self.conv_size//2),
                # nn.ReLU(inplace=True),
                nn.Linear(self.conv_size, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 6 * self.num_slots),
                nn.Sigmoid()
                )
        
        # coordinate patching trick
        coord_map = create_coord_buffer(self.image_shape).repeat(self.batch_size,1,1,1)
        self.register_buffer('coord_map_const', coord_map)
        
        constrain_add = torch.tensor([[[self.min, 0., -1], [0., self.min, -1]]])
        self.register_buffer('constrain_add', constrain_add)
        constrain_mult = torch.tensor([[[self.max-self.min, 0., 2.], [0., self.max-self.min, 2.]]])
        self.register_buffer('constrain_mult', constrain_mult)

    def theta_restrict(self, theta):
        return F.mse_loss(theta * self.theta_mask, self.theta_mean)

    def forward(self, x):
        inp = torch.cat([x, self.coord_map_const], 1)
        conv = self.detection_network(inp)
        conv = conv.view(-1, self.conv_size)
        theta = self.theta_regression(conv)
        theta = theta.view(-1, self.num_slots, 2, 3)
        if self.constrain_theta:
            theta = (theta * self.constrain_mult) + self.constrain_add
        return theta


class MaskNet(nn.Module):
    """
    Attention network for mask prediction
    """
    def __init__(self, conf, in_channels=7, num_blocks=None):
        super().__init__()
        self.conf = conf
        if not num_blocks:
            num_blocks = conf.num_blocks
        self.unet = UNet(num_blocks=num_blocks,
                         in_channels=in_channels,
                         out_channels=1,
                         channel_base=conf.channel_base)

    def forward(self, x):
        logits = self.unet(x)
        alpha = torch.sigmoid(logits)
        return alpha


class SpatialAutoEncoder(nn.Module):
    """
    Spatial transformation and reconstruction auto encoder
    """
    def __init__(self, conf):
        super().__init__()
        self.prior = conf.latent_prior
        self.fg_sigma = conf.fg_sigma
        self.bg_sigma = conf.bg_sigma
        self.patch_shape = conf.patch_shape
        self.image_shape = conf.image_shape

        self.encoding_network = EncoderNet(conf)
        self.decoding_network = DecoderNet(conf)
        self.mask_network = MaskNet(conf)
        # self.spatial_network = SpatialLocalizationNet(conf)
        
    def forward(self, x, theta):
        # calculate spatial position for object from joint mask
        # and image
        grid = F.affine_grid(theta, torch.Size((x.size()[0], 1, *self.patch_shape)))

        # get patch from theta and grid
        x_patch = transform(x, grid, theta)
        
        # generate object mask for patch
        mask = self.mask_network(x_patch)

        # concatenate x and new mask
        encoder_input = torch.cat([x_patch, mask.detach()], 1)

        # generate latent embedding of the patch
        mean, sigma = self.encoding_network(encoder_input)
        z, kl_z = differentiable_sampling(mean, sigma, self.prior)

        decoded = self.decoding_network(z)
        
        # seperate patch into img and scope
        patch_reconstruction = decoded[:, :3, :, :]
        mask_pred = decoded[:, 3:, :, :]
        
        # calculate mask prediction error
        kl_mask_pred = kl_mask(mask.detach(), mask_pred)

        # transform all components into original space
        x_reconstruction = invert(patch_reconstruction, theta, self.image_shape)
        mask = invert(mask, theta, self.image_shape)
        
        scope = x[:, 6:, :, :]
        mask = mask * scope
        p_x = reconstruction_likelihood(
                x[:, :3], 
                x_reconstruction,
                mask,
                self.fg_sigma)
        scope = scope - mask

        return x_reconstruction, mask, scope, z, kl_z, p_x, kl_mask_pred


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
        conf.component_latent_dim = 3
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
    
    def init_bias(self, images):
        with torch.no_grad():
            self.dec_net.network[-1].bias.zero_()
            norm = len(images.dataset)
            append = torch.zeros((1, self.image_shape[0], self.image_shape[1]))
            for image_batch in images:
                for image in image_batch[0]:
                    fill = torch.cat([image, append], 0).cuda()
                    self.dec_net.network[-1].bias += torch.mean(fill.view(-1, 1))
            self.dec_net.network[-1].bias /= norm


class FCBackgroundModel(nn.Module):
    """
    Background encoder model working on a single fully connected layer
    """
    def __init__(self, conf):
        super().__init__()
        
        self.image_shape = conf.image_shape
        image_shape_flattened = self.image_shape[0] * self.image_shape[1] * 4
        self.net = nn.Linear(1, image_shape_flattened, bias=False)

    def init_bias(self, images):
        with torch.no_grad():
            self.net.weight.zero_()
            norm = len(images.dataset)
            append = torch.zeros((1, self.image_shape[0], self.image_shape[1]))
            for image_batch in images:
                for image in image_batch[0]:
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
    def __init__(self, conf):
        super().__init__()

        self.bg_sigma = conf.bg_sigma
        self.fg_sigma = conf.fg_sigma

        self.spatial_vae = SpatialAutoEncoder(conf)
        # self.background_model = BackgroundEncoder(conf)
        # self.background_model = VAEBackgroundModel(conf)
        self.background_model = FCBackgroundModel(conf)
        # self.mask_background = MaskNet(conf, 6, 4)
        self.spatial_localization_net = SpatialLocalizationNet(conf)

        self.num_slots = conf.num_slots

        self.beta = conf.beta
        self.gamma = conf.gamma

        self.running = 0

    def init_background_weights(self, images):
        self.background_model.init_bias(images)
    
    def forward(self, x):
        # initialize arrays for visualization
        masks = []
        latents = []
        mask_preds = []

        # initialize loss components
        scope = torch.ones_like(x[:, :1])
        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        kl_zs = torch.zeros_like(loss)
        kl_masks = torch.zeros_like(loss)
        p_x_loss = torch.zeros_like(x)
        
        background, _ = self.background_model(x, scope)
        background = background[:, :3, :, :]

        # get all thetas at once
        inp = torch.cat([x, (x-background).detach()], 1)
        thetas = self.spatial_localization_net(inp)

        # construct the patchwise shaping of the model
        for i in range(self.num_slots):
            theta = thetas[:, i]
            inp = torch.cat([x, ((x-background) - total_reconstruction).detach(), scope], 1)
            x_recon, mask, scope, z, kl_z, p_x, kl_m = self.spatial_vae(inp, theta)
            kl_zs += kl_z
            p_x_loss += p_x
            kl_masks += kl_m
            total_reconstruction += mask * x_recon
            
            # save for visualization
            masks.append(mask)
            latents.append(z)
        
        total_reconstruction += background * scope

        # calculate reconstruction error
        p_x = reconstruction_likelihood(x, 
                background,
                (1 - torch.sum(torch.cat(masks, 1), 1, True)), 
                self.bg_sigma, 
                self.running)
        p_x_loss += p_x

        # calculate the final loss
        assert not torch.any(torch.isnan(p_x_loss)), 'p_x nan'
        assert not torch.any(torch.isnan(kl_zs)), 'kl z nan'
        assert not torch.any(torch.isnan(kl_masks)), 'kl mask nan'
        loss = -p_x_loss.mean([1,2,3]) + self.beta * kl_zs + self.gamma * kl_masks

        # torchify all outputs
        masks.insert(0, scope)
        masks = torch.cat(masks, 1)
        latents = torch.cat(latents, 1)

        self.running += 1

        # currently missing is the mask reconstruction loss
        return {'loss': loss,
                'reconstructions': total_reconstruction,
                'reconstruction_loss': p_x_loss,
                'masks': masks,
                'latents': latents,
                'mask_loss': kl_masks,
                'kl_loss': kl_zs}
