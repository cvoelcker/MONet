import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from datasets import Atari
import spatial_monet

data = sys.argv[2]


def show_tensor(datum):
    img = np.transpose(datum[0][0].numpy(), (1,2,0))
    plt.imshow(img)
    plt.show()


def main():
    conf = pickle.load(open('model_conf', 'rb'))
    monet = spatial_monet.MaskedAIR(conf.model_config).cuda()
    monet = torch.nn.DataParallel(monet, device_ids=[0])
    monet.load_state_dict(torch.load(conf.run_config.checkpoint_file))
    shape = (128,128)
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    dataframe = Atari(data, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataframe,
                                              batch_size=8)

    with torch.no_grad():
        ret = monet.forward(next(iter(trainloader))[0])
        for k in ret.keys():
            ret[k] = ret[k].cpu()
        pickle.dump(ret, open('first_batch.save', 'wb'))


if __name__ == '__main__':
    main()
