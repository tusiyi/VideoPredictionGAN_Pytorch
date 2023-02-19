import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np


class KITTIDataset(Dataset):
    def __init__(self, image_dir, image_channel=3, image_size=[128, 160], seq_len=9, num_past_frames=8, inter=1):
        super(KITTIDataset, self).__init__()
        self.image_dir = image_dir
        self.image_channel = image_channel
        self.image_size = image_size
        self.seq_len = seq_len
        self.num_past_frames = num_past_frames
        self.inter = inter  # frame interval of videos
        self.data = self.get_sequence(image_dir, seq_len)  # Image object
        self.transform = transforms.Compose([VidRandomHorizontalFlip(0.5), VidToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data[item]  # (seq_len, H, W, 3)
        if self.transform is not None:
            seq = self.transform(seq)
        past_frames = seq[:self.num_past_frames]
        future_frames = seq[self.num_past_frames:]
        # 归一化和转为tensor已经在self.transform的VidToTensor中完成
        past_frames = past_frames.float()
        future_frames = future_frames.float()
        return past_frames, future_frames

    def get_sequence(self, root, seq_len):
        """
        read image sequence in image root directory
        :param root: image sequence root directory, each sub-directory contains png images of one sequence
        :param seq_len: sequence length for one sample
        :return:
        """
        dirs = os.listdir(root)  # sub-directory
        data = []
        n = 0
        for d in dirs:
            image_name = os.listdir(os.path.join(root, d))
            image_name.sort()
            imgs = []
            for name in image_name:
                path = os.path.join(root, d, name)
                img = Image.open(path)
                img = img.resize((self.image_size[1], self.image_size[0]))
                imgs.append(img)  # H, W, 3
            # frame interval larger than 1
            if self.inter > 1:
                imgs = [imgs[i] for i in range(len(imgs)) if i % self.inter == 0]
            # none overlap video clips
            for i in range(0, len(imgs) // seq_len):
                data.append(imgs[i * seq_len: (i + 1) * seq_len])
        return data  # List[List[Image.Image]],  (N, seq_len, Image对象)


class CaltechDataset(Dataset):
    def __init__(self, image_dir, image_channel=3, image_size=[128, 160], seq_len=9, num_past_frames=8, inter=1):
        super(CaltechDataset, self).__init__()
        self.image_dir = image_dir
        self.image_channel = image_channel
        self.image_size = image_size
        self.seq_len = seq_len
        self.num_past_frames = num_past_frames
        self.inter = inter
        self.data = self.get_sequence(image_dir, seq_len, inter=inter)
        self.transform = VidToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data[item]  # (seq_len, H, W, 3)
        if self.transform is not None:
            seq = self.transform(seq)
        past_frames = seq[:self.num_past_frames]
        future_frames = seq[self.num_past_frames:]
        # 归一化和转为tensor已经在self.transform的VidToTensor中完成
        past_frames = past_frames.float()
        future_frames = future_frames.float()
        return past_frames, future_frames

    def get_sequence(self, root, seq_len, inter=1):
        """
        read image sequence in image root directory
        :param root: image sequence root directory, each sub-directory contains png images of one sequence
        :param seq_len: sequence length for one sample
        :param inter: interval for frames(e.g. when inter=2, only select 0, 2, 4, ... these frames)
        :return:
        """
        dirs = os.listdir(root)  # sub-directories
        data = []
        # n = 0
        for d in dirs:
            image_name = os.listdir(os.path.join(root, d))
            image_name.sort()
            imgs = []
            for name in image_name:
                path = os.path.join(root, d, name)
                img = Image.open(path)
                img = img.resize((self.image_size[1], self.image_size[0]))
                imgs.append(img)  # H, W, 3
            if inter > 1:
                imgs = [imgs[i] for i in range(len(imgs)) if i % inter == 0]
            for i in range(0, len(imgs) // seq_len):
                data.append(imgs[i * seq_len: (i + 1) * seq_len])
        return data  # List[List[Image.Image]],  (N, seq_len, Image对象)


def get_dataloader(dataset, data_dir, batch_size=8, train_val_ratio=1.0, seq_len=9, num_past_frames=8, inter=1,
                   image_size=[128, 160]):
    if dataset == 'KITTI':
        # train and val set
        train_whole_set = KITTIDataset(data_dir, image_channel=3, image_size=image_size,
                                       seq_len=seq_len, num_past_frames=num_past_frames, inter=inter)
        if train_val_ratio == 1.0:
            return DataLoader(train_whole_set, batch_size=batch_size, shuffle=True, num_workers=2), None
        else:
            n_train = int(len(train_whole_set) * train_val_ratio)
            n_val = len(train_whole_set) - n_train
            train_set, val_set = random_split(train_whole_set, lengths=[n_train, n_val],
                                              generator=torch.Generator().manual_seed(1))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)  # , drop_last=True)
            return train_loader, val_loader
    elif dataset == 'Caltech' or dataset == 'BDD':  # BDD100K dataset can use the same dataset class as well
        # for testing
        test_set = CaltechDataset(data_dir, image_channel=3, image_size=image_size,
                                  seq_len=seq_len, num_past_frames=num_past_frames, inter=inter)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
        return test_loader
    else:
        raise ValueError('Not implemented dataset yet.')


# Video transform classes
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert 0 <= p <= 1, "invalid flip probability"
        self.p = p

    def __call__(self, clip):
        """clip: List[Image.Image] """
        if np.random.rand() < self.p:
            # flip 是随机的，每一个batch有可能flip也有可能不会
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip


class VidToTensor(object):
    def __call__(self, clip):
        """
        clip: List[Image.Image]
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim=0)
        # print(clip.shape)
        return clip
