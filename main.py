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

import spatial_monet.monet as model
import spatial_monet.util.datasets as datasets
import spatial_monet.util.experiment_config as experiment_config

from spatial_monet.util.handlers import TensorboardHandler
from spatial_monet.util.train_util import save_results_after_training, numpify, visualize_masks

import spatial_monet.spatial_monet as spatial_monet


def run_training(monet, conf, trainloader, parallel=True):
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
            if parallel:
                images = images.cuda()
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(monet.parameters(), 10)
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
        monet.module.beta = sigmoid(0 - 10 + epoch)

    print('training done')
    # save_results_after_training(monet, trainloader, conf.checkpoint_file)
    print('saved final results')


def masked_air_experiment():
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
    parallel=False
    model_conf_dict = experiment_config.record_type_to_dict(model_conf)
    print(model_conf_dict)
    monet = spatial_monet.MaskedAIR(model_conf_dict)
    if run_conf.parallel:
        parallel = True
        device_id = 0
        torch.cuda.set_device(device_id)
        if run_conf.summarize:
            b_s = model_conf.batch_size
            model_conf.batch_size = b_s
        monet = monet.cuda()
        sum([param.nelement() for param in monet.parameters()])
        monet = nn.DataParallel(monet, device_ids=[device_id])
    run_training(monet, run_conf, trainloader, parallel)


if __name__ == '__main__':
    masked_air_experiment()
