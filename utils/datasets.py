import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

# from utils.augmentations import horisontal_flip
from utils.augmentations import SSDAugmentation,BaseTransform
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=320):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor

        img = cv2.imread(img_path)
        Augmentation = BaseTransform(320)
        img, _, _ = Augmentation(img)
        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255
        # img = transforms.ToTensor()(Image.open(img_path))
        # # Pad to square resolution
        # img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=320, augment=True, multiscale=True, normalized_labels=False):
        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()
        # folder_path = "E:/TF_Code/PyTorch-YOLOv3-master/PyTorch-YOLOv3-master/data/image"
        self.img_files = sorted(glob.glob("%s/*.*" % list_path))
        self.label_files = [
            path.replace("image", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # print(img_path)
        img = cv2.imread(img_path)
        # Handle images with less than three channels

        # if len(img.shape) != 3:
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        h, w, _ = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # targets = None
        if os.path.exists(label_path):
            boxes =np.loadtxt(label_path).reshape(-1, 5)
            # print(boxes)
            # Extract coordinates for unpadded + unscaled image

            boxes[:, 1] =  (boxes[:, 1])/w
            boxes[:, 2] =  (boxes[:, 2])/h
            boxes[:, 3] =  (boxes[:, 3])/w
            boxes[:, 4] =  (boxes[:, 4])/h



        if self.augment:
            boxes = np.array(boxes)
            Augmentation=SSDAugmentation(self.img_size)
            img, boxe, labels = Augmentation(img, boxes[:, 1:], boxes[:, 0])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            img =torch.from_numpy(img).permute(2, 0, 1)
            img = img/255
            _,h1, w1= img.shape

            target = torch.from_numpy(np.hstack((np.expand_dims(labels, axis=1),boxe)))

            x1 = w1*target[:, 1]
            y1 = h1*target[:, 2]
            x2 = w1*target[:, 3]
            y2 = h1*target[:, 4]

            target[:, 1] = ((x1 + x2) / 2) / w1
            target[:, 2] = ((y1 + y2) / 2) / h1
            target[:, 3] = (x2 - x1) / w1
            target[:, 4] = (y2 - y1) / h1

            targets = torch.zeros((len(target), 6))
            targets[:, 1:] = target

        else:

            boxes = np.array(boxes)
            Augmentation=BaseTransform(self.img_size)
            img, boxe, labels = Augmentation(img, boxes[:, 1:], boxes[:, 0])
            img = img[:, :, (2, 1, 0)]
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = img / 255
            _, h1, w1 = img.shape

            target = torch.from_numpy(np.hstack((np.expand_dims(labels, axis=1), boxe)))

            x1 = w1 * target[:, 1]
            y1 = h1 * target[:, 2]
            x2 = w1 * target[:, 3]
            y2 = h1 * target[:, 4]

            target[:, 1] = ((x1 + x2) / 2) / w1
            target[:, 2] = ((y1 + y2) / 2) / h1
            target[:, 3] = (x2 - x1) / w1
            target[:, 4] = (y2 - y1) / h1

            targets = torch.zeros((len(target), 6))
            targets[:, 1:] = target

        # Apply augmentations
        # if self.augment:
        #     if np.random.random() < 0.5:
        #         img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":


    train_path = '/home/ecust/gmy/PyTorch-YOLOv3(steel)/PyTorch-YOLOv3-master/data/NEU-DET/train/image'
    valid_path = '/home/ecust/gmy/PyTorch-YOLOv3(steel)/PyTorch-YOLOv3-master/data/NEU-DET/valid/image'
    dataset = ListDataset(train_path, augment=True, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    batch_iterator = iter(dataloader)
    _,images, targets = next(batch_iterator)
    print(images.size(), targets)


    # a=sorted(glob.glob("%s/*.*" % train_path))
    # label_files = [
    #     path.replace("image", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
    #     for path in a
    # ]
    # print(len(a))
    # print(len(label_files))
    # for i in label_files:
    #     boxes = torch.from_numpy(np.loadtxt(i).reshape(-1, 5))
    #     print(i,boxes.shape)
