# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import re

import torch
import torch.optim
import argparse
import albumentations as A
import my_albumentations as M

from tqdm import tqdm
from config import parse_test_config, parse_train_config, DEVICE, read_yaml_config
from metrics import MetricFunction, print_single_error
from datetime import datetime as dt
from model import Model, LossFunction
from test import test
from general import images_to_device, init_weights, save_checkpoint, load_checkpoint
from dataset import create_dataloader


def train_one_epoch(model, dataloader, loss_fn, metric_fn, solver, epoch_idx):
    loop = tqdm(dataloader, position=0, leave=True)

    for _, images in enumerate(loop):
        images = images_to_device(images, DEVICE)
        (left_img, right_img, left_normal, _) = images

        predictions = model(left_img, right_img)
        loss = loss_fn(predictions, left_normal)
        metric_fn.evaluate(predictions, left_normal)

        model.zero_grad()
        loss.backward()
        solver.step()

        loop.set_postfix(loss=loss_fn.show(), epoch=epoch_idx)
    loop.close()

def train(config=None, config_test=None):
    torch.backends.cudnn.benchmark = True

    config = parse_train_config() if not config else config

    transform = A.Compose(
        [
            M.MyRandomResizedCrop(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                M.MyOpticalDistortion(p=0.3),
                M.MyGridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'right_img': 'image',
            'left_normal': 'normal',
            'right_normal': 'normal',
        }
    )

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH,
                                      batch_size=config.BATCH_SIZE, transform=transform,
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    model = Model()
    model.apply(init_weights)
    solver = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config.LEARNING_RATE, betas=config.BETAS,
                              eps=config.EPS, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=config.MILESTONES, gamma=config.GAMMA)
    model = model.to(DEVICE)

    loss_fn = LossFunction()

    epoch_idx = 0
    if config.CHECKPOINT_FILE and config.LOAD_MODEL:
        epoch_idx, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    output_dir = os.path.join(config.OUT_PATH, re.sub("[^0-9a-zA-Z]+", "-", dt.now().isoformat()))

    for epoch_idx in range(epoch_idx, config.NUM_EPOCHS):
        metric_fn = MetricFunction(config.BATCH_SIZE)

        model.train()
        train_one_epoch(model, dataloader, loss_fn, metric_fn, solver, epoch_idx)
        print_single_error(epoch_idx, loss_fn.show(), metric_fn.show())
        lr_scheduler.step()

        if config.TEST:
            test(model, config_test)
        if config.SAVE_MODEL:
            save_checkpoint(epoch_idx, model, output_dir)

    if not config.TEST:
        test(model, config_test)
    if not config.SAVE_MODEL:
        save_checkpoint(epoch_idx, model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train', type=str, default="train.yaml", help='train config file')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()

    config_train = parse_train_config(read_yaml_config(opt.train))
    config_test = parse_test_config(read_yaml_config(opt.test))

    train(config_train, config_test)
