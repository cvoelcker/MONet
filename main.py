import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
import spatial_monet.util.datasets as datasets
import spatial_monet.util.experiment_config as experiment_config

from spatial_monet.util.train_util import run_training

import spatial_monet.spatial_genesis as spatial_monet
import spatial_monet.genesis as genesis


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
                              transform=transform,
                              source_type='dill')
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=run_conf.batch_size,
                                              shuffle=True, num_workers=8)
    parallel=False
    model_conf_dict = model_conf._asdict()
    print(model_conf_dict)
    monet = genesis.GENESIS(**model_conf_dict)
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
    run_dict = experiment_config.record_type_to_dict(run_conf)
    run_training(monet, trainloader, **run_dict, beta_overwrite = 1.)


def debug_experiment():
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
    trainset = datasets.BlackWhite(100, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=run_conf.batch_size,
                                              shuffle=True, num_workers=8)
    parallel=False
    model_conf_dict = experiment_config.record_type_to_dict(model_conf)
    print(model_conf_dict)
    monet = genesis.GENESIS(**model_conf_dict)
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
    run_dict = run_conf._asdict()
    run_training(monet, trainloader, **run_dict, beta_overwrite = 1.)


if __name__ == '__main__':
    # debug_experiment()
    masked_air_experiment()

