import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler
import torch.utils.data as data
from RotNetModel1 import RotNetMulti
from RotNetModel1 import RotNetMultiPretrained
import argparse
import torch.nn as nn
import vggcifar
import samplerMulti2
from custom_datasets import *
import vggcifarpretrained
from solverMulti import Solver
import arguments

def caltech101_transformer():
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])

def main(args):
    print("Seed 101")

    caltech101_exists = os.path.exists(os.path.join(args.data_path, 'caltech101'))

    if args.dataset == 'caltech101':
        args.num_val = 914
        args.num_images = 8232
        args.budget = 411
        args.initial_budget = 822
        args.num_classes = 102

        all_indices = set(np.arange(args.num_images))
        test_indices = random.sample(list(all_indices), 822)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)
        all_indices = np.setdiff1d(list(all_indices), test_indices)

        train_dataset = Caltech101(os.path.join(args.data_path, 'caltech101'))
        rot_train_dataset = rot_Caltech101(os.path.join(args.data_path, 'caltech101'))

        test_dataloader = data.DataLoader(train_dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False, num_workers=0)

        print("Random sampling")
        all_indices = set(np.arange(args.num_images))
        val_indices = random.sample(list(all_indices), args.num_val)
        all_indices = np.setdiff1d(list(all_indices), val_indices)

        initial_indices = random.sample(list(all_indices), args.initial_budget)
        sampler = data.sampler.SubsetRandomSampler(initial_indices)
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)

        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True, num_workers=4)
        val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler, batch_size=args.batch_size, drop_last=False, num_workers=4)
        rot_dataloader = data.DataLoader(rot_train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True, num_workers=4)
        rot_val_dataloader = data.DataLoader(rot_train_dataset, sampler=val_sampler, batch_size=args.batch_size, drop_last=True, num_workers=4)

        solver = Solver(args, test_dataloader)
        samplerRot = samplerMulti2.RotSampler(args.budget, args)
        splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

        current_indices = list(initial_indices)
        num_img1 = len(current_indices)

        accuracies = []

        for split in splits:
            task_model = vggcifar.vgg16_bn(num_classes=args.num_classes)
            rotNet1 = RotNetMulti(num_classes=args.num_classes, num_rotations=4)

            # Ensure everything runs on CPU
            task_model = task_model.cuda()
            rotNet1 = rotNet1.cuda()

            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
            remain_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
            unlabeled_dataloader = data.DataLoader(train_dataset, sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False, num_workers=4)
            rot_unlabeled_dataloader = data.DataLoader(rot_train_dataset, sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False, num_workers=4)

            acc = solver.train(querry_dataloader, val_dataloader, task_model, unlabeled_dataloader, num_img1)
            print('Final accuracy of Task network with {}% of data is: {:.2f}'.format(int(split * 100), acc))

            accuracies.append(acc)
            new_random = random.sample(list(remain_indices), args.budget)
            current_indices = list(current_indices) + list(new_random)
            sampler = data.sampler.SubsetRandomSampler(current_indices)

            querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True, num_workers=0)
            rot_dataloader = data.DataLoader(rot_train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True, num_workers=0)

            num_img1 = len(current_indices)
            torch.save(accuracies, os.path.join(args.out_path, args.log_name))

        torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
