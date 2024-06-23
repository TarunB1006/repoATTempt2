import os
os.environ['PYTHONHASHSEED'] = str(100)

import random
import numpy as np
import torch
import torch.utils.data as data
import argparse
from torchvision import datasets, transforms
from PIL import Image
import vggcifar
import vggcifarpretrained
from solverMulti import Solver
import samplerMulti2
from custom_datasets import *

def caltech101_transformer():
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])

def main(args):
    print("Seed 101")
    print(args)

    # Check if Caltech-101 dataset is already downloaded
    caltech101_exists = os.path.exists(os.path.join(args.data_path, 'caltech101'))

    if args.dataset == 'caltech101':
        test_dataloader = DataLoader(
            datasets.ImageFolder(os.path.join(args.data_path, 'caltech101', 'test'), transform=caltech101_transformer()),
            batch_size=args.batch_size, shuffle=False, num_workers=0)

        train_dataset = Caltech101(os.path.join(args.data_path, 'caltech101'))
        rot_train_dataset = rot_Caltech101(os.path.join(args.data_path, 'caltech101'))

        args.num_val = 914
        args.num_images = 8232
        args.budget = 411
        args.initial_budget = 822
        args.num_classes = 102

    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    val_indices = random.sample(list(all_indices), args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # Dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                        batch_size=args.batch_size, drop_last=True, num_workers=0)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
                                     batch_size=args.batch_size, drop_last=False, num_workers=0)
    rot_dataloader = data.DataLoader(rot_train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True, num_workers=0)
    rot_val_dataloader = data.DataLoader(rot_train_dataset, sampler=val_sampler, batch_size=args.batch_size, drop_last=True, num_workers=0)

    print("Running on cuda")

    solver = Solver(args, test_dataloader)
    samplerRot = samplerMulti2.RotSampler(args.budget, args)
    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)
    num_img1 = len(current_indices)

    accuracies = []

    for split in splits:
        task_model = vggcifar.vgg16_bn(num_classes=args.num_classes)
        task_model = task_model.cuda()

        # Get unlabeled indice dataloader
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                                               sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False, num_workers=0)
        rot_unlabeled_dataloader = data.DataLoader(rot_train_dataset, sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False, num_workers=0)

        # Train task network on current labeled pool
        acc = solver.train(querry_dataloader, val_dataloader, task_model, unlabeled_dataloader, rot_dataloader, rot_val_dataloader, rot_unlabeled_dataloader, samplerRot, split)
        accuracies.append(acc)

    print(f"Final Accuracies: {accuracies}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with RoT training on Caltech-101')

    parser.add_argument('--data_path', default='data', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    parser.add_argument('--dataset', default='caltech101', type=str, choices=['caltech101'],
                        help='choose Caltech-101 dataset')
    
    # Add additional arguments here
    parser.add_argument('--out_path', type=str, help='output path')
    parser.add_argument('--train_epochs', type=int, help='number of training epochs')
    parser.add_argument('--log_name', type=str, help='name of the log file')
    parser.add_argument('--optim_task', type=str, help='optimizer for task model')
    parser.add_argument('--scheduler_task', type=str, help='scheduler for task model')
    parser.add_argument('--lr_task', type=float, help='learning rate for task model')
    parser.add_argument('--optim_Rot', type=str, help='optimizer for rotation model')
    parser.add_argument('--scheduler_Rot', type=str, help='scheduler for rotation model')
    parser.add_argument('--lr_rot', type=float, help='learning rate for rotation model')
    parser.add_argument('--rot_train_epochs', type=int, help='number of training epochs for rotation model')
    parser.add_argument('--lambda_kl', type=float, help='weight for KL divergence')
    parser.add_argument('--train_loss_weightClassif', type=float, help='weight for classification loss during training')
    parser.add_argument('--val_loss_weightClassif', type=float, help='weight for classification loss during validation')
    parser.add_argument('--lambda_rot', type=float, help='weight for rotation loss')
    parser.add_argument('--val_loss_weightRotation', type=float, help='weight for rotation loss during validation')
    parser.add_argument('--train_loss_weightRotation', type=float, help='weight for rotation loss during training')
    parser.add_argument('--valtype', type=str, help='type of validation (e.g., loss)')
    parser.add_argument('--samplebatch_size', type=int, help='batch size for sampling')
    parser.add_argument('--lambda_div', type=float, help='weight for diversity loss')

    args = parser.parse_args()
    main(args)
