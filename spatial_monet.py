import torch
import torch.nn as nn
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
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(18, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class TransformerNet(nn.Module):
    def __init(self, height, width, latent_dim, z_dim):
        super().__init__()
        self.height = height
        self.width = width

        # little bit of a bad name, this is actually the spatial network "viewsize"
        self.latent_dim_x = latent_dim[0]
        self.latent_dim_y = latent_dim[1]

        self.z_dim = 

        self.latent_size = torch.Size((batch_size, 4, self.latent_dim_x, self.latent_dim_y))
        
        # inverts both 2d picture and 1d masks
        self.original_size_3 = torch.Size((batch_size, 3, height, width))
        self.original_size_1 = torch.Size((batch_size, 1, height, width))

        # taken from the demo
        # TODO: adapt network architecture to problem specific implementation
        self.localization = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
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
        self.encoder_network = EncoderNet(*latent_dim, 2 * z_dim)
        self.decoder_network = DecoderNet(*latent_dim)

    # Spatial transformer network forward function
    def stn(self, x):
        size = x.size()
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        print(theta)

        grid = F.affine_grid(theta, self.latent_size)
        x = F.grid_sample(x, grid)

        return x, theta

    def invert_stn(self, x, theta, i):
        # i manages whether a 3d or a 1d object is reconstructed
        inverse_theta = torch.pinverse(theta)
        if i == 1:
            grid = F.affine_grid(inverse_theta, self.original_size_1)
        elif i == 3:
            grid = F.affine_grid(inverse_theta, self.original_size_3)
        x = F.grid_sample(x, grid)
        return x
    
    def encoder_network(self, x, mask):
        complete = torch.stack([x, mask], dim=-1)
        x_embedded = self.encoder_network(complete)
        # seperates the x vector into mu and sigma
        latent_mean = x_embedded[:, :self.z_dim]
        latent_sigma = x_embedded[:, self.z_dim:]
        return latent_mean, latent_sigma

    def decoder_network(self, z):
        return self.decoder_network(z)
        

    def forward(self, x, scope):
        complete = torch.stack([x, scope], -1)
        # transform the input to attend to a part of the network
        x_zoom, theta = self.stn(complete)

        # seperate scope and picture
        x_zoom = x_zoom[:,:,:,:3]
        scope_zoom = x_zoom[:,:,:,3:4]

        # build a mask over the cropped picture frame
        logits = mask_network(x_zoom, scope_zoom)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = alpha[:, 0:1]
        new_scope = alpha[:, 1:2]

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
        latent_vae = means + dist_0.sample()
        q_z = dist.log_prob(latent_vae)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)

        # decode and multiply by mask
        x_recon = self.decoder_network(latent_vae)

        # project reconstruction and mask back into original space
        mask = scope * self.invert_stn(mask, theta, 1)
        new_scope = self.invert_stn(scope, theta, 1)
        new_scope = scope * new_scope
        
        x_recon = self.invert_stn(x_recon, theta, 3) * mask
        
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])

        return {'reconstruction': x, 
                'mask': mask, 
                'scope': new_scope, 
                'latent': (latent_mean, latent_sigma, latent_vae, theta),
                'kl_latent': kl_z,
                'p_x': p_x}


class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.transformer_network = TransformerNet(height, width, self.conf.latent_dim)
        self.beta = 0.5
        self.gamma = 0.25

    def forward(self, x, pass_latent=False):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        latents = []

        total_reconstruction = torch.zeros_like(x)

        loss = torch.zeros_like(x[:, 0, 0, 0])
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i in range(self.conf.num_slots-1):
            transformer_input = torch.stack([total_mask, x], dim=-1)
            res = self.transformer_network(transformer_input, scope)
            total_reconstruction += res['reconstruction']
            masks.append(res['mask'])
            scope = res['scope']
            latents.append['latent']
            
            # joined compete loss with hyperparameters
            loss += -res['p_x'] + self.beta * res['kl_latent']
            p_xs += -res['p_x']
            kl_zs += res['kl_latent']

        # mask loss
        masks = torch.cat(masks, 1)
        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        
        loss += self.gamma * kl_masks

        return {'loss': loss,
                'masks': masks,
                'latents': latents,}



        # Old code for Karls Monet

        # masks.append(scope)
        # loss = torch.zeros_like(x[:, 0, 0, 0])
        # mask_preds = []
        # full_reconstruction = torch.zeros_like(x)
        # p_xs = torch.zeros_like(loss)
        # kl_zs = torch.zeros_like(loss)
        # zs = []
        # for i, mask in enumerate(masks):
        #     z, kl_z = self.__encoder_step(x, mask)
        #     if pass_latent:
        #         zs.append(z)
        #     sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
        #     p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
        #     mask_preds.append(mask_pred)
        #     loss += -p_x + self.beta * kl_z
        #     p_xs += -p_x
        #     kl_zs += kl_z
        #     full_reconstruction += mask * x_recon

        # masks = torch.cat(masks, 1)
        # tr_masks = torch.transpose(masks, 1, 3)
        # q_masks = dists.Categorical(probs=tr_masks)
        # q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        # kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        # kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        # loss += self.gamma * kl_masks
        # if pass_latent:
        #     return {'loss': loss,
        #             'masks': masks,
        #             'latent': zs}
        # return {'loss': loss,
        #         'masks': masks,
        #         'reconstructions': full_reconstruction}

    # def __encoder_step(self, x, mask):
    #     encoder_input = torch.cat((x, mask), 1)
    #     q_params = self.encoder(encoder_input)
    #     means = torch.sigmoid(q_params[:, :16]) * 6 - 3
    #     sigmas = torch.sigmoid(q_params[:, 16:]) * 3
    #     dist = dists.Normal(means, sigmas)
    #     dist_0 = dists.Normal(0., sigmas)
    #     z = means + dist_0.sample()
    #     q_z = dist.log_prob(z)
    #     kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
    #     kl_z = torch.sum(kl_z, 1)
    #     return z, kl_z

    # def __decoder_step(self, x, z, mask, sigma):
    #     decoder_output = self.decoder(z)
    #     x_recon = torch.sigmoid(decoder_output[:, :3])
    #     mask_pred = decoder_output[:, 3]
    #     dist = dists.Normal(x_recon, sigma)
    #     p_x = dist.log_prob(x)
    #     p_x *= mask
    #     p_x = torch.sum(p_x, [1, 2, 3])
    #     return p_x, x_recon, mask_pred




## old code


# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
# 
# 
# class UNet(nn.Module):
#     def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
#         super().__init__()
#         self.num_blocks = num_blocks
#         self.down_convs = nn.ModuleList()
#         cur_in_channels = in_channels
#         for i in range(num_blocks):
#             self.down_convs.append(double_conv(cur_in_channels,
#                                                channel_base * 2**i))
#             cur_in_channels = channel_base * 2**i
# 
#         self.tconvs = nn.ModuleList()
#         for i in range(num_blocks-1, 0, -1):
#             self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
#                                                   channel_base * 2**(i-1),
#                                                   2, stride=2))
# 
#         self.up_convs = nn.ModuleList()
#         for i in range(num_blocks-2, -1, -1):
#             self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))
# 
#         self.final_conv = nn.Conv2d(channel_base, out_channels, 1)
# 
#     def forward(self, x):
#         intermediates = []
#         cur = x
#         for down_conv in self.down_convs[:-1]:
#             cur = down_conv(cur)
#             intermediates.append(cur)
#             cur = nn.MaxPool2d(2)(cur)
# 
#         cur = self.down_convs[-1](cur)
# 
#         for i in range(self.num_blocks-1):
#             cur = self.tconvs[i](cur)
#             cur = torch.cat((cur, intermediates[-i -1]), 1)
#             cur = self.up_convs[i](cur)
# 
#         return self.final_conv(cur)
# 
# 
# class AttentionNet(nn.Module):
#     def __init__(self, conf):
#         super().__init__()
#         self.conf = conf
#         self.unet = UNet(num_blocks=conf.num_blocks,
#                          in_channels=4,
#                          out_channels=2,
#                          channel_base=conf.channel_base)
# 
#     def forward(self, x, scope):
#         inp = torch.cat((x, scope), 1)
#         logits = self.unet(inp)
#         alpha = torch.softmax(logits, 1)
#         # output channel 0 represents alpha_k,
#         # channel 1 represents (1 - alpha_k).
#         mask = scope * alpha[:, 0:1]
#         new_scope = scope * alpha[:, 1:2]
#         return mask, new_scope
# 
# class EncoderNet(nn.Module):
#     def __init__(self, width, height):
#         super().__init__()
#         self.convs = nn.Sequential(
#             nn.Conv2d(4, 32, 3, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, 3, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, stride=2),
#             nn.ReLU(inplace=True)
#         )
#         for i in range(4):
#             width = (width - 1) // 2
#             height = (height - 1) // 2
#         self.mlp = nn.Sequential(
#             nn.Linear(64 * width * height, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 32)
#         )
# 
#     def forward(self, x):
#         x = self.convs(x)
#         x = x.view(x.shape[0], -1)
#         x = self.mlp(x)
#         return x
# 
# class DecoderNet(nn.Module):
#     def __init__(self, height, width):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.convs = nn.Sequential(
#             nn.Conv2d(18, 32, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 4, 1),
#         )
#         ys = torch.linspace(-1, 1, self.height + 8)
#         xs = torch.linspace(-1, 1, self.width + 8)
#         ys, xs = torch.meshgrid(ys, xs)
#         coord_map = torch.stack((ys, xs)).unsqueeze(0)
#         self.register_buffer('coord_map_const', coord_map)
# 
#     def forward(self, z):
#         z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
#         coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
#         inp = torch.cat((z_tiled, coord_map), 1)
#         result = self.convs(inp)
#         return result

