import torch
import torch.nn as nn
import torch.distributions as dists

import torchvision

import spatial_monet.util.network_util as net_util


class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = net_util.UNet(num_blocks=conf.num_blocks,
                         in_channels=4,
                         out_channels=2,
                         channel_base=conf.channel_base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope


class EncoderNet(nn.Module):
    def __init__(self, width, height):
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
            nn.Linear(256, 32)
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
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8,
                                                       self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.beta = 0.25
        self.gamma = 0.25

    def forward(self, x, pass_latent=False):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf.num_slots - 1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        zs = []
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            if pass_latent:
                zs.append(z)
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        masks = torch.cat(masks, 1)
        mask_preds = torch.stack(mask_preds, 3)
        tr_masks = torch.transpose(masks, 1, 3)
        # hotfix for wrong transpose
        tr_masks = torch.transpose(tr_masks, 1, 2)

        print(masks.size())
        print(mask_preds.size())
        print(tr_masks.size())
        exit()

        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=mask_preds)
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        if pass_latent:
            return {'loss': loss,
                    'masks': masks,
                    'latent': zs}
        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction}

    def build_image_graph(self, x):
        """
        Builds the graph representation of an image from one forward pass
        of the model
        """
        loss, masks, embeddings = self.forward(x, pass_latent=True).values()

        grid = net_util.center_of_mass(masks[:, 1:])
        gridX = grid[..., :1] - grid[..., :1].permute(0, 2, 1)
        gridY = grid[..., 1:] - grid[..., 1:].permute(0, 2, 1)
        grid = torch.stack([gridX, gridY], -1)

        grid_embeddings = embeddings.unsqueeze(2)
        grid_interactions = (grid_embeddings - grid_embeddings.permute(0, 2, 1,
                                                                       3)) / 2
        grid_embeddings = grid_interactions + grid_embeddings

        return torch.cat([grid_embeddings, grid], -1), loss

    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :16]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 16:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred
