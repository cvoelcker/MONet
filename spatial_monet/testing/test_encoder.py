import torch

from recordtype import recordtype
import pickle
from tqdm import tqdm

from spatial_monet import EncoderNet, DecoderNet, differentiable_sampling, reconstruction_likelihood


conf_template = recordtype(
        'MaskedAIRModelConfiguration', 
        [
            ('component_latent_dim', 64),
            ('background_latent_dim', 1),
            ('latent_prior', 1.0),
            ('patch_shape', (32, 32)),
            ('channel_base', 32),
            ('batch_size', 8),
            ('num_slots', 8),
            ('beta', 1.),
            ('gamma', 1.),
            ('constrain_theta', True),
            ])


class Model(torch.nn.Module):

    def __init__(self, _conf):
        super(Model, self).__init__()
        self.enc = EncoderNet(_conf, 4).cuda()
        self.dec = DecoderNet(_conf).cuda()

    def forward(self, x):
        mean, sigma = self.enc(x)
        z, kl = differentiable_sampling(mean, sigma, 1)
        x = self.dec(z)
        return x, kl


def main():
    di = 2
    torch.cuda.set_device(di)
    conf = conf_template()
    model = Model(conf).cuda()

    train_set = pickle.load(open('small_images', 'rb'))
    zeros = torch.zeros_like(train_set)

    augmented_set = torch.cat([train_set, zeros, zeros, zeros, zeros], 0)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, num_workers=4, shuffle=False)
    augmented_loader = torch.utils.data.DataLoader(augmented_set, batch_size = 8, num_workers=4, shuffle=True)
    model = torch.nn.DataParallel(model, device_ids=[di])

    optim = torch.optim.Adam(model.parameters(), lr=7e-4)
    
    all_losses = []

    for epoch in tqdm(list(range(20))):
        for image in tqdm(augmented_loader):
            optim.zero_grad()
            mask = torch.ones_like(image)
            inp = torch.cat([image, mask[:, :1]], 1)
            recon, kls = model(inp)
            recon_loss = -reconstruction_likelihood(image.cuda(), recon[:, :3], mask.cuda(), 0.1)
            total_loss = torch.mean(recon_loss + kls)
            total_loss.backward()
            optim.step()
            all_losses.append(total_loss.detach().cpu().item())
    
    final_reconstructions = []
    for image in tqdm(train_loader):
        mask = torch.ones_like(image)
        inp = torch.cat([image, mask[:, :1]], 1)
        recon, kls = model(inp)
        final_reconstructions.append(recon.detach().cpu().numpy())

    pickle.dump(final_reconstructions, open('reconstructions_with_black', 'wb'))
    pickle.dump(all_losses, open('losses_with_black', 'wb'))


if __name__ == '__main__':
    main()
