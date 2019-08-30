import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
import numpy as np
import visdom

import sys
import os
from datetime import datetime
import tqdm
import pickle

import model
import spatial_monet
import datasets
import config
import experiment_config
from handlers import TensorboardHandler


def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks(imgs, masks, recons, vis):
    # print('recons min/max', recons.min().item(), recons.max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    colors.extend([(c[0]//8, c[1]//8, c[2]//8) for c in colors])

    masks = np.argmax(masks, 1)
    seg_maps = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]
    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])


def run_training(monet, conf, trainloader):
    vis = visdom.Visdom(env = conf.visdom_env, port=8456)
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        # monet = torch.load('the_whole_fucking_thing')
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        monet.module.init_background_weights(trainloader)
        print('Initialized parameters')

    tbhandler = TensorboardHandler('logs/', conf.visdom_env)
    # optimizer = optim.RMSprop(monet.parameters(), lr=conf.step_size)
    optimizer = optim.Adam(monet.parameters(), lr=conf.step_size)
    all_gradients = []

    beta_max = monet.module.beta
    gamma_max = monet.module.gamma
    gamma_increase = gamma_max

    sigmoid = lambda x: 1/(1+np.exp(-x))

    monet.module.beta = sigmoid(0 - 10)
    monet.module.gamma = gamma_increase

    print(monet.module.beta)
    torch.autograd.set_detect_anomaly(False)

    for epoch in tqdm.tqdm(list(range(conf.num_epochs))):
        running_loss = 0.0
        mask_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        epoch_loss = []
        epoch_reconstruction_loss = []
        for i, data in enumerate(tqdm.tqdm(trainloader), 0):
            images, counts = data
            if images.shape[0] < conf.batch_size:
                continue
            images = images.cuda()
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            # all_norms = []
            # for n, p in monet.named_parameters():
            #     if not np.all(np.isfinite(p.data.detach().cpu().numpy())):
            #         print(n)
            #         exit()
            #     all_norms.append((n, p.grad.data.clone().norm().item(), p.data.cpu().clone().numpy()))
            norm = torch.nn.utils.clip_grad_norm_(monet.parameters(), 10)
            # if not np.isfinite(norm):
            #     for pair in all_norms:
            #         print(pair)
            #     for n, p in monet.named_parameters():
            #         print()
            #         print(n)
            #         print(p.grad.data.norm())

            assert np.isfinite(norm), f'norm nan {loss}'
            optimizer.step()
            running_loss += loss.detach().item()
            mask_loss += output['mask_loss'].mean().detach().item()
            kl_loss += output['kl_loss'].mean().detach().item()
            recon_loss += output['reconstruction_loss'].mean().detach().item()

            epoch_loss.append(loss.detach().item())
            epoch_reconstruction_loss.append(torch.mean(output['reconstruction_loss']).detach().item())

            if i % conf.vis_every == conf.vis_every-1:
                gradients = [(n, p.grad) for n, p in monet.named_parameters()]
                gradients = [(g[0], (torch.mean(g[1]).item() if torch.is_tensor(g[1]) else g[1])) for g in gradients]
                all_gradients.append(gradients)
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / conf.vis_every))
                visualize_masks(numpify(images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]), 
                                vis)
                handler_data = {'tf_logging': {
                        'loss': running_loss/conf.vis_every,
                        'kl_loss': kl_loss/conf.vis_every,
                        'px_loss': recon_loss/conf.vis_every,
                        'mask_loss': mask_loss/conf.vis_every},
                        'step': conf.vis_every,
                        }
                tbhandler.run(monet, handler_data)
                running_loss = 0.0
                mask_loss = 0.0
                recon_loss = 0.0
                kl_loss = 0.0
        torch.save(monet.state_dict(), conf.checkpoint_file)
        monet.module.beta = sigmoid(0 - 10 + epoch * 20 /(conf.num_epochs))
        # pickle.dump(all_gradients, open('gradients.save', 'wb'))
        # monet.module.beta += beta_increase
        # monet.module.gamma += gamma_increase

    print('training done')
    torch.save(monet, 'the_whole_fucking_thing')

def sprite_experiment():
    conf = config.sprite_config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    trainset = datasets.Sprites(conf.data_dir, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=2)
    monet = model.Monet(conf, 64, 64).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

def clevr_experiment():
    conf = config.clevr_config
    # Crop as described in appendix C
    crop_tf = transforms.Lambda(lambda x: transforms.functional.crop(x, 29, 64, 192, 192))
    drop_alpha_tf = transforms.Lambda(lambda x: x[:3])
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf,
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = datasets.Clevr(conf.data_dir,
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    monet = model.Monet(conf, 128, 128).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

def spatial_transform_experiment():
    conf = config.spatial_transform_config
    shape = (128,128)
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    if conf.reshape:
        shape = (128,128)
        transform = transforms.Compose([transforms.Lambda(lambda x: transforms.functional.crop(x, 16, 0, 170, 160)),
                                        transforms.Resize(shape),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.float()),
                                       ])
    trainset = datasets.Atari(conf.data_dir,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    if conf.parallel:
        torch.cuda.set_device(1)
        monet = spatial_monet.Monet(conf, 
                                    *shape).cuda()
        # summary(monet, trainset[0][0].shape)
        monet = nn.DataParallel(monet, device_ids=[1])
    else:
        monet = spatial_monet.Monet(conf, 
                                    *shape)
        # print(conf.batch_size)
        # print(trainset[0][0].shape)
        # summary(monet, trainset[0][0].shape, device="cpu")
        #monet = nn.DataParallel(monet, device_ids=[0])
    run_training(monet, conf, trainloader)


def atari_experiment():
    conf = config.atari_config
    shape = (256,128)
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    if conf.reshape:
        shape = (128,128)
        transform = transforms.Compose([transforms.Lambda(lambda x: transforms.functional.crop(x, 16, 0, 170, 160)),
                                        transforms.Resize(shape),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.float()),
                                       ])
    trainset = datasets.Atari(conf.data_dir,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    if conf.parallel:
        torch.cuda.set_device(3)
        # monet = model.Monet(conf, *shape).cuda()
        # summary(monet, trainset[0][0].shape)
        monet = nn.DataParallel(monet, device_ids=[1,2])
    else:
        monet = model.Monet(conf, *shape)
        #summary(monet, trainset[0][0].shape)
        monet = nn.DataParallel(monet, device_ids=[0])

    run_training(monet, conf, trainloader)

    monet = model.Monet(conf, *shape).cuda()
    summary(monet, trainset[0][0].shape)


def spatial_transform_experiment():
    conf = config.spatial_transform_config
    shape = (128,128)
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    if conf.reshape:
        shape = (128,128)
        transform = transforms.Compose([transforms.Lambda(lambda x: transforms.functional.crop(x, 16, 0, 170, 160)),
                                        transforms.Resize(shape),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.float()),
                                       ])
    trainset = datasets.Atari(conf.data_dir,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    if conf.parallel:
        torch.cuda.set_device(0)
        monet = spatial_monet.Monet(conf, 
                                    *shape).cuda()
        # summary(monet, trainset[0][0].shape)
        monet = nn.DataParallel(monet, device_ids=[1])
    else:
        monet = spatial_monet.Monet(conf, 
                                    *shape)
        # print(conf.batch_size)
        # print(trainset[0][0].shape)
        summary(monet, trainset[0][0].shape, device="cpu")
        # monet = nn.DataParallel(monet, device_ids=[0])
    run_training(monet, conf, trainloader)


def reimplementation_experiment():
    conf = experiment_config.parse_args_to_config(sys.argv[1:])
    run_conf = conf.run_config
    model_conf = conf.model_config
    shape = model_conf.image_shape
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    if run_conf.reshape:
        shape = (128,128)
        transform = transforms.Compose([transforms.Lambda(lambda x: transforms.functional.crop(x, 16, 0, 170, 160)),
                                        transforms.Resize(shape),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.float()),
                                       ])
    trainset = datasets.Atari(run_conf.data_dir,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=run_conf.batch_size,
                                              shuffle=True, num_workers=8)
    if run_conf.parallel:
        pickle.dump(conf, open('model_conf', 'wb'))
        device_id = 0
        torch.cuda.set_device(device_id)
        if run_conf.summarize:
            b_s = model_conf.batch_size
            model_conf.batch_size = b_s
        monet = spatial_monet.MaskedAIR(model_conf).cuda()
        sum([param.nelement() for param in monet.parameters()])
        monet = nn.DataParallel(monet, device_ids=[device_id])
    run_training(monet, run_conf, trainloader)


if __name__ == '__main__':
    # clevr_experiment()
    # sprite_experiment()
    # atari_experiment()
    # spatial_transform_experiment()
    reimplementation_experiment()
