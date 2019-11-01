from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":


    os.makedirs("ground_truth", exist_ok=True)
    classes = load_classes("data/coco.names")  # Extracts class labels from file

    path = 'data/NEU-DET/valid/labels'
    folder_path = 'data/samples'
    files = sorted(glob.glob("%s/*.*" % folder_path))
    # name = [i.split('/')[-1].split('.')[0].strip()  for i in files]
    label = [os.path.join(path, i.split('/')[-1].split('.')[0].strip()+'.txt') for i in files]
    # print(files)
    # print(label)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    for img_path,label_path in zip(files,label):

        boxes = np.loadtxt(label_path).reshape(-1, 5)

        imgs.append(img_path)
        img_detections.append(boxes)

    # print(imgs)
    # print(img_detections)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        # print(detections)
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            # detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            # unique_labels = detections[:, 0]
            # n_cls_preds = len(unique_labels)
            # bbox_colors = random.sample(colors, n_cls_preds)
            for cls, x1, y1, x2, y2 in detections:
                # print( x1, y1, x2, y2)

                print("\t+ Label: %s" % (classes[int(cls)]))

                box_w = x2 - x1
                box_h = y2 - y1

                color = colors[int(cls)]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"ground_truth/{filename}.jpg", bbox_inches="tight", pad_inches=0.0)
        plt.close()

