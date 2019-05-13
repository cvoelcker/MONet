import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

import torchvision


class EncoderNet(nn.Module):
    def __init__(self, width, height, z_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )
        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, z_dim)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, height, width, z_dim):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height)
        xs = torch.linspace(-1, 1, self.width)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height, self.width)
        # print(z_tiled.size())
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        # print(inp.size())
        result = self.convs(inp)
        return result


class TransformerNet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf

        self.height = height
        self.width = width

        # little bit of a bad name, this is actually the spatial network "viewsize"
        self.latent_dim_x, self.latent_dim_y = conf.latent_dim
        self.z_dim = conf.z_dim
        self.batch_size = conf.batch_size

        self.latent_size = torch.Size((self.batch_size, 4, self.latent_dim_x, self.latent_dim_y))
        
        # inverts both 2d picture and 1d masks
        self.original_size_3 = torch.Size((self.batch_size, 3, height, width))
        self.original_size_1 = torch.Size((self.batch_size, 1, height, width))

        # taken from the demo
        # TODO: adapt network architecture to problem specific implementation
        #       dimensionality should fit, but this might be completely wrong
        self.localization = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=3),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=3),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # encoder network for the VAE component
        # z_dim is the number of Gaussian, so
        # it is replicated, ones for mean and ones
        # for sigma
        self.encoder_net = EncoderNet(*conf.latent_dim, 2 * conf.z_dim)
        self.decoder_net = DecoderNet(*conf.latent_dim, conf.z_dim)
        self.mask_net = nn.Sequential(
                nn.Conv2d(4,32,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32,32,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32,32,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                )
        self.mlp = nn.Sequential(
                nn.Linear(32 * self.latent_dim_x * self.latent_dim_y, self.latent_dim_x * self.latent_dim_y))

        self.theta_append = torch.Tensor([0., 0., 1.]).view(1,1,-1).repeat(self.batch_size,1,1)
        print(self.theta_append)

    # Spatial transformer network forward function
    def stn(self, x):
        size = x.size()
        xs = self.localization(x)
        xs = xs.view(-1, 64)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, self.latent_size)
        x = F.grid_sample(x, grid)

        return x, theta

    def invert_stn(self, x, theta, i):
        # i manages whether a 3d or a 1d object is reconstructed
        
        # really ugly hack, since the pseudinverse is not implemented for
        # batch wise computation
        inverse_theta = torch.cat([theta, self.theta_append], dim=1).inverse()
        if i == 1:
            grid = F.affine_grid(inverse_theta[:, :2, :], self.original_size_1)
        elif i == 3:
            grid = F.affine_grid(inverse_theta[:, :2, :], self.original_size_3)
        x = F.grid_sample(x, grid)
        return x
    
    def encoder_network(self, x, mask):
        complete = torch.cat([x, mask], dim=1)
        x_embedded = self.encoder_net(complete)
        # seperates the x vector into mu and sigma
        latent_mean = x_embedded[:, :self.z_dim]
        latent_sigma = x_embedded[:, self.z_dim:]
        return latent_mean, latent_sigma

    def decoder_network(self, z):
        # print(z.size())
        return self.decoder_net(z)

    def mask_network(self, x, scope):
        # print(x.size())
        # print(scope.size())
        complete = torch.cat([x, scope], dim=1)
        mask = self.mask_net(complete)
        mask_dense = mask.view(-1,32 * self.latent_dim_x * self.latent_dim_y)
        mask_dense = self.mlp(mask_dense)
        return mask_dense.view(-1,1,self.latent_dim_x, self.latent_dim_y)

    def forward(self, x, scope, i):
        complete = torch.cat([x, scope], 1)
        # transform the input to attend to a part of the network
        x_zoom, theta = self.stn(complete)

        # seperate scope and picture
        x_zoom = x_zoom[:,:3,:,:]
        scope_zoom = x_zoom[:,2:4,:,:]

        # build a mask over the cropped picture frame
        logits = self.mask_network(x_zoom, scope_zoom)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = alpha[:, 0:1]
        new_scope = alpha[:, 1:2]

        # start of VAE
        # build a latent for the mask embedding
        latent_mean, latent_sigma = self.encoder_network(x_zoom, mask * scope_zoom)

        # vae prior loss
        # TODO: there is a mistake here, since the mask does not
        # interact with the VAE, but it should given the code
        # in the paper. I bleieve it's not that bad, since
        # we have the spatial attention network, which already 
        # represents a mask, but we should check that
        # TODO: probably fixed by above
        dist = dists.Normal(latent_mean, latent_sigma)
        dist_0 = dists.Normal(0., latent_sigma)
        latent_vae = latent_mean + dist_0.sample()
        q_z = dist.log_prob(latent_vae)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)

        # decode and multiply by mask
        x_recon = self.decoder_network(latent_vae)
        mask_pred = x_recon[:, 3:]
        x_recon = x_recon[:, :3]

        # project reconstruction and mask back into original space
        mask = scope * self.invert_stn(mask, theta, 1)
        mask_pred = self.invert_stn(mask_pred, theta, 1)
        new_scope = self.invert_stn(scope, theta, 1)
        new_scope = scope * new_scope
        
        x_recon = self.invert_stn(x_recon, theta, 3) * mask
        
        self.results_dict = {'reconstruction': x, 
                'mask': mask, 
                'scope': new_scope, 
                'latent': (latent_mean, latent_sigma, latent_vae, theta),
                'kl_latent': kl_z,
                'mask_pred': mask_pred}
        return x_recon


class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.transformer_network = TransformerNet(conf, height, width)
        self.beta = 0.5
        self.gamma = 0.25
        self.width = width
        self.height = height

    def forward(self, x):
        # print(x.size())
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        latents = []
        mask_preds = []

        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i in range(self.conf.num_slots-1):
            x_recon = self.transformer_network(x, scope, i)
            res = self.transformer_network.results_dict
            total_reconstruction += res['reconstruction']
            masks.append(res['mask'])
            scope = res['scope']
            latents.append(res['latent'])
            mask_preds.append(res['mask_pred'].view(-1, self.width, self.height))
            print(res['mask'].size())
            
        # TODO: missing implementation for the final stage
        #       where the rest of the scope is simply used
        sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
        dist = dists.Normal(total_reconstruction, sigma)
        p_x = dist.log_prob(x)
        p_x = torch.sum(p_x, [1, 2, 3])
        
        loss += -p_x + self.beta * torch.sum(kl_zs)

        # mask loss
        masks = torch.cat(masks, 1)
        # hotfix for wrong transpose
        tr_masks = torch.transpose(masks, 1, 3)
        tr_masks = torch.transpose(tr_masks, 1, 2)
        
        mask_preds = torch.stack(mask_preds, 3)

        print(masks.size())
        print(tr_masks.size())
        print(mask_preds.size())
        
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=mask_preds)
        
        print(q_masks.probs.size())
        print(q_masks_recon.probs.size())

        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks

        return {'loss': loss,
                'masks': masks,
                'latents': latents,}

