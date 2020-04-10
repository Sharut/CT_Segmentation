from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, img_mean=0, img_var=1, mask_mean=0, mask_var=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.img_mean = img_mean
        self.img_var = img_var
        self.mask_mean = mask_mean
        self.mask_var = mask_var

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def internet_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255


        return img_trans

    def preprocess(cls, np_img, scale, mean, var):
        w, h = np_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        
        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, axis=2)
        
        # Clip image and HWC to CHW
        img_trans = np_img.transpose((2, 0, 1))
        img_trans = np.clip(img_trans, -1000, 1000)
        # Normalise
        img_trans = (img_trans-mean)/var
        
        

        # transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((mean,),(var,))
        #         ])

        # img_tensor = TF.to_tensor(img_trans)
        # print(img_tensor.shape)
        # print("Mean and std ", mean, var)
        # normalized_tensor = TF.normalize(img_tensor, mean=mean, std=var)
        return img_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0])
        img = np.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, self.img_mean, self.img_var)
        mask = self.preprocess(mask, self.scale, self.mask_mean, self.mask_var)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
        # return {'image': img, 'mask': mask}

