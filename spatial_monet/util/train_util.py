import os

import numpy as np
import torch
from torch import nn
import pickle
import visdom

from tqdm import tqdm

from spatial_monet.util.handlers import TensorboardHandler


def save_results_after_training(model, data, save_location):
    image_save = []
    mask_save = []
    recon_save = []
    for i, images in enumerate(tqdm(data), 0):
        img, counts = images
        img = img.cuda()
        with torch.no_grad():
            output = model(img)
            image_save.append(numpify(img))
            mask_save.append(numpify(output['masks']))
            recon_save.append(numpify(output['reconstructions']))
    image_save = np.concatenate(image_save, 0)
    mask_save = np.concatenate(mask_save, 0)
    recon_save = np.concatenate(recon_save, 0)
    pickle.dump(image_save[:500], open(save_location + '_images.pkl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(mask_save[:500], open(save_location + '_masks.pkl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(recon_save[:500], open(save_location + '_recon.pkl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


def numpify(tensor):
    return tensor.cpu().detach().numpy()


def visualize_masks(imgs, masks, recons, vis):
    # print('recons min/max', recons.min().item(), recons.max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
              (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0] // 2, c[1] // 2, c[2] // 2) for c in colors])
    colors.extend([(c[0] // 4, c[1] // 4, c[2] // 4) for c in colors])
    colors.extend([(c[0] // 8, c[1] // 8, c[2] // 8) for c in colors])

    masks = np.argmax(masks, 1)
    seg_maps = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]
    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])


def run_training(monet, trainloader, step_size=7e-4, num_epochs=1,
                 batch_size=8, visdom_env='default', vis_every=50,
                 load_parameters='false', checkpoint_file='default',
                 parallel=True, initialize=True, tbhandler=None, 
                 beta_overwrite=None, **kwargs):
    print(batch_size)
    # vis = visdom.Visdom(env=visdom_env, port=8456)
    if load_parameters and os.path.isfile(checkpoint_file):
        # monet = torch.load('the_whole_fucking_thing')
        monet.load_state_dict(torch.load(checkpoint_file))
        print('Restored parameters from', checkpoint_file)
    elif initialize:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        monet.module.init_background_weights(trainloader)
        print('Initialized parameters')

    if tbhandler is None:
        tbhandler = TensorboardHandler('../master_thesis_code/logs/', visdom_env)
    # optimizer = optim.RMSprop(monet.parameters(), lr=conf.step_size)
    optimizer = torch.optim.Adam(monet.parameters(), lr=step_size)
    all_gradients = []

    if beta_overwrite is None:
        beta_max = monet.module.beta
        gamma_max = monet.module.gamma
        gamma_increase = gamma_max

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        monet.module.beta = sigmoid(0 - 10)
        monet.module.gamma = gamma_increase
    else:
        monet.module.beta = beta_overwrite

    torch.autograd.set_detect_anomaly(False)

    for epoch in tqdm(list(range(num_epochs))):
        running_loss = 0.0
        mask_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        epoch_loss = []
        epoch_reconstruction_loss = []
        for i, data in enumerate(tqdm(trainloader), 0):
            images = data
            if images.shape[0] < batch_size:
                continue
            if parallel:
                images = images.cuda()
            
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(monet.parameters(), 5)
            optimizer.step()
            running_loss += loss.detach().item()
            mask_loss += output['mask_loss'].mean().detach().item()
            kl_loss += output['kl_loss'].mean().detach().item()
            recon_loss += output['reconstruction_loss'].mean().detach().item()

            epoch_loss.append(loss.detach().item())
            epoch_reconstruction_loss.append(
                torch.mean(output['reconstruction_loss']).detach().item())

            assert not torch.isnan(torch.sum(loss))

            if i % vis_every == vis_every - 1:
                gradients = [(n, p.grad) for n, p in monet.named_parameters()]
                gradients = [(g[0], (
                    torch.mean(g[1]).item() if torch.is_tensor(g[1]) else g[
                        1])) for g in gradients]
                all_gradients.append(gradients)
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / conf.vis_every))
                # visualize_masks(numpify(images[:8]),
                #                 numpify(output['masks'][:8]),
                #                 numpify(output['reconstructions'][:8]),
                #                 vis)
                handler_data = {'tf_logging': {
                    'loss': running_loss / vis_every,
                    'kl_loss': kl_loss / vis_every,
                    'px_loss': recon_loss / vis_every,
                    'mask_loss': mask_loss / vis_every},
                    'step': vis_every * batch_size,
                }
                tbhandler.run(monet, handler_data)
                running_loss = 0.0
                mask_loss = 0.0
                recon_loss = 0.0
                kl_loss = 0.0
        torch.save(monet.state_dict(), checkpoint_file)
        if beta_overwrite is None:
            monet.module.beta = sigmoid(0 - 10 + epoch)

    print('training done')
    # save_results_after_training(monet, trainloader, conf.checkpoint_file)
    print('saved final results')
