import numpy as np
import torch
import pickle

from tqdm import tqdm


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
    mask_save  = np.concatenate(mask_save, 0)
    recon_save = np.concatenate(recon_save, 0)
    pickle.dump(image_save[:500], open(save_location + '_images.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(mask_save[:500], open(save_location + '_masks.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(recon_save[:500], open(save_location + '_recon.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


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
