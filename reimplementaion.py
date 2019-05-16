import collections

import torch

import torch.nn as nn
import torch.nn.functional as F

from model import import UNet


ModelConfiguration = collections.namedtuple(
        'ModelConfiguration', 
        [
            'component_latent_dim', 
            'background_latent_dim', 
            'latent_prior', 
            'patch_shape', 
            'image_shape',
            'num_layers',
            'bg_sigma',
            'num_blocks',
            'channel_base'
            ])

def create_coord_buffer(patch_size):
    ys = torch.linspace(-1, 1, self.patch_shape[0])
    xs = torch.linspace(-1, 1, self.patch_shape[1])
    coord_map = torch.stack((ys, xs)).unsqueeze(0)
    return coord_map


def differentiable_sampling(mean, sigma, prior_sigma):
    dist = dists.Normal(mean, sigma)
    dist_0 = dists.Normal(0., sigma)
    z = latent_mean + dist_0.sample()
    kl_z = dists.kl_divergence(dist, dists.Normal(0., prior_sigma))
    kl_z = torch.sum(kl_z, 1)
    return z, kl_z


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
            nn.ReLU(inplace=True)
            nn.MaxPool2d(2, stride=2),
            )
        conv_size = 64 * self.patch_shape[0]/(2**4) * self.patch_shape[1]/(2**4)
        self.mlp = nn.Sequqential(
                nn.Linear(conv_size, 2 * self.latent_dim),
                nn.ReLU(inplace=True))
        self.mean_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.network(x)
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
        coord_map = generate_coord_buffer(self.patch_shape)
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


class SpatialLocalizationNet(nn.model):
    """
    Attention network for object localization
    """
    def __init__(self, conf):
        super().__init__()

        self.image_shape = conf.image_shape
        self.patch_shape = conf.patch_shape
        self.batch_size = conf.batch_size
        
        self.detection_network = nn.Sequential(
                # block 1
                nn.Conv2d(3 + 1 + 2, 16, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 32, width/2, length/2

                # block 2
                nn.Conv2d(32, 32, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 64, width/4, length/4

                # block 3
                nn.Conv2d(64, 64, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                # output size = 128, width/8 length/8
                )
        conv_size = 64 * self.image_shape[0]/8 * self.image_shape[1]/8
        self.theta_regression = nn.Sequential(
                nn.Linear(conv_size, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 6)
                nn.Tanh(inplace=True)
                )
        
        # coordinate patching trick
        coord_map = generate_coord_buffer(self.image_shape)
        self.register_buffer('coord_map_const', coord_map)
        
        # helper sizes for grid generation
        self.latent_size = torch.Size((self.batch_size, 4, *self.patch_shape))
        self.original_size_3 = torch.Size((self.batch_size, 3, *self.image_shape))
        self.original_size_1 = torch.Size((self.batch_size, 1, *self.image_shape))
        
        theta_append = torch.Tensor([0., 0., 1.]).view(1,1,-1).repeat(self.batch_size,1,1)
        self.register_buffer('theta_append', theta_append)

    def forward(self, x):
        inp = torch.cat([x, coord_map], 1)
        conv = self.network(inp)
        theta = self.theta_regression(conv)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, self.latent_size)
        return grid, theta

    def transform(self, x, grid, theta):
        x = F.grid_sample(x, grid)
        return x

    def invert(self, x, theta):
        inverse_theta = torch.cat([theta, self.theta_append], dim=1).inverse()
        i = x.size()[1]
        if i == 1:
            grid = F.affine_grid(inverse_theta[:, :2, :], self.original_size_1)
        elif i == 3:
            grid = F.affine_grid(inverse_theta[:, :2, :], self.original_size_3)
        x = F.grid_sample(x, grid)
        return x


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
        self.encoding_network = EncoderNet(conf)
        self.decoding_network = DecoderNet(conf)
        self.mask_network = MaskNet(conf)
        self.spatial_network = SpatialLocalizationNet(conf)

    def forward(self, x):
        # calculate spatial position for object from joint mask
        # and image
        theta, grid = self.spatial_network(x)

        # get patch from theta and grid
        x_patch = self.spatial_network.transform(x, grid, theta)

        # generate object mask for patch
        # TODO: this seems highly questionable, check again
        logits = self.mask_network(x_patch)
        alpha = torch.softmax(logits, 1)
        old_scope = x_patch[:, 1:, :, :]
        mask = old_scope * alpha[:, :1]
        new_scope = old_scope * alpha[:, 1:]

        # concatenate x and new mask
        encoder_input = torch.cat(x_zoom, mask)

        # generate latent embedding of the patch
        mean, sigma = self.encoding_network(encoder_input)
        z, kl_z = differentiable_sampling(mean, sigma, self.prior)

        decoded = self.decoding_network(z)
        
        # seperate patch into img and scope
        patch_reconstruction = decoded[:, :3, :, :] * mask
        mask_pred = decoded[:, 3:, :, :]

        # transform all components into original space
        x_reconstruction = self.spatial_network.invert(x, theta)
        mask_transformation = self.spatial_network.invert(mask, theta)
        mask_prediction = self.spatial_network.invert(mask_pred, theta)
        
        scope = x[:, 3:, :, :]
        scope -= mask_reconstruction

        return x_reconstruction, mask_transformation, mask_prediction, scope, z, kl_z


class BackgroundEncoder(nn.Module):
    """
    Background encoder model which captures the rest of the picture
    TODO: Currently completely stupid
    """
    def __init__(self, conf):
        super.__init__()

        self.image_shape = conf.image_shape


    def forward(self, x):
        loss = torch.zeros_like(x[:, 0, 0, 0])
        return torch.zeros_like(x), loss


class MaskedAIR(nn.Module):
    """
    Full model for image reconstruction and mask prediction
    """
    def __init__(self, conf):
        super().__init__()

        self.num_layers = conf.num_layers
        self.bg_sigma = conf.bg_sigma

        self.spatial_vae = SpatialAutoEncoder(conf)
        self.background_model = BackgroundEncoder(conf)
    
    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        latents = []
        mask_preds = []

        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        kl_zs = torch.zeros_like(loss)

        # construct the patchwise shaping of the model
        for i in range(self.conf.num_slots-1):
            inp = torch.cat([x, scope], 1)
            x_recon, mask, mask_pred, scope, z, kl_z = self.spatial_vae(inp)
            total_reconstruction += mask * x_recon
            masks.append(mask)
            mask_preds.append(mask_pred)
            latents.append(z)
            kl_zs += kl_z
        
        # compute background
        inp_background = torch.cat([x, scope], 1)
        background, kl_z = self.background_model(inp_background)
        kl_zs += kl_z
        total_reconstruction += background * scope

        # compute reconstruction loss
        dist = dists.Normal(total_reconstruction, self.bg_sigma)
        p_x = dist.log_prob(x)
        p_x = torch.sum(p_x, [1, 2, 3])
        
        loss += -p_x + self.beta * kl_zs
        
        # currently missing is the mask reconstruction loss
        return {'loss': loss,
                'reconstructions': total_reconstruction,
                'masks': masks,
                'latents': latents,}
