import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transform import *
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
import csv
from glob import glob 


def load_data_from_zhnogshan():
    root = './data/poly_detection'
    img_path = f'{root}/images/*'
    mask_path = f'{root}/masks/*'

    img_path_list, label_path_lst = sorted(glob(img_path)), sorted(glob(mask_path))

    return img_path_list, label_path_lst


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, args, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if args.dataset == "Train_on_ZhongshanandKvasirandDB":
            zhong_img_lst,zhong_mask_lst=load_data_from_zhnogshan()
            self.images += zhong_img_lst
            self.gts += zhong_mask_lst
            
        if args.dataset == "Train_on_Zhongshan":
            zhong_img_lst,zhong_mask_lst=load_data_from_zhnogshan()
            self.images = zhong_img_lst
            self.gts = zhong_mask_lst

        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
                   Resize((self.trainsize, self.trainsize)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #RandomCrop((self.trainsize, self.trainsize)),
                   ToTensor(),

               ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        data = {'image': image, 'label': gt}
        data = self.transform(data)
        return data

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(args, image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset(args, image_root, gt_root, trainsize=trainsize)

    train_sampler = DistributedSampler(dataset, shuffle=True)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=multiprocessing.Pool()._processes,  
                                  pin_memory=True)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.gt_transform =  transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
