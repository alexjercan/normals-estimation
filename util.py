# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def load_normal(path):
    img = exr2normal(path)  # 3 channel normal
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)

    img = img / 255

    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def exr2normal(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img[img > 1] = 1
    img[img < 0] = 0
    
    img = np.array(img).astype(np.float32).reshape(img.shape[0], img.shape[1], -1)

    return img


def plot_predictions(images, predictions, paths):
    left_images, _ = images
    normal_ps = predictions
    left_images = left_images.cpu().numpy()
    normal_ps = normal_ps.cpu().numpy()

    for left_img, normal_p, path in zip(left_images, normal_ps, paths):
        left_img = left_img.transpose(1, 2, 0)
        normal = normal_p.transpose(1, 2, 0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(path)
        ax1.axis('off')
        ax1.imshow(left_img)
        ax2.axis('off')
        ax2.imshow(normal)
        plt.show()

def save_predictions(predictions, paths):
    normal_ps = predictions
    normal_ps = normal_ps.cpu().numpy()

    for normal_p, path in zip(normal_ps, paths):
        normal = normal_p.transpose(1, 2, 0)

        normal_path = str(Path(path).with_suffix(".exr"))

        cv2.imwrite(normal_path, normal)
        
        plt.axis('off')
        plt.imshow(normal)
        plt.savefig(str(Path(path).with_suffix(".png")))
