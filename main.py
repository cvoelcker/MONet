import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
import numpy as np
import visdom

import os
from datetime import datetime
import tqdm

import model
import spatial_monet
import datasets
import config

vis = visdom.Visdom(env = 'rl_pictures_{}'.format(datetime.now().strftime('%Y-%m-%d')))


def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks(imgs, masks, recons):
    # print('recons min/max', recons.min().item(), recons.max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])

def run_training(monet, conf, trainloader):
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in tqdm.tqdm(range(conf.num_epochs)):
        running_loss = 0.0
        epoch_loss = []
        for i, data in enumerate(tqdm.tqdm(trainloader), 0):
            images, counts = data
            images = images.cuda()
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss.append(loss.item())

            if i % conf.vis_every == conf.vis_every-1:
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / conf.vis_every))
                running_loss = 0.0
                visualize_masks(numpify(images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]))

        torch.save(monet.state_dict(), conf.checkpoint_file)
        # print(np.mean(epoch_loss))


    print('training done')

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
    torch.cuda.set_device(3)
    monet = spatial_monet.Monet(conf, *shape).cuda()
    summary(monet, trainset[0][0].shape)
    if conf.parallel:
        monet = nn.DataParallel(monet, device_ids=[0])
    run_training(monet, conf, trainloader)

    monet = spatial_monet.Monet(conf, *shape).cuda()
    summary(monet, trainset[0][0].shape)

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
    torch.cuda.set_device(3)
    monet = spatial_monet.Monet(conf, 
                                *shape).cuda()
    summary(monet, trainset[0][0].shape)
    if conf.parallel:
        monet = nn.DataParallel(monet, device_ids=[0])
    run_training(monet, conf, trainloader)

    monet = model.Monet(conf, *shape).cuda()
    summary(monet, trainset[0][0].shape)

if __name__ == '__main__':
    # clevr_experiment()
    # sprite_experiment()
    # atari_experiment()
    spatial_transform_experiment()

