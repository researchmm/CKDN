from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
from PIL import Image
import numpy as np
import random

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if build_class_idx and not subdirs:
            class_to_idx[label] = None
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            phase,
            load_bytes=False,
            transform=None):

        val_types = ['_00_02','_00_03','_00_04','_00_05','_00_06','_00_07','_01_09','_01_10','_01_11','_01_12','_01_13','_01_14','_01_15','_03_21','_03_22','_03_23','_03_05','_03_06','_03_07','_03_08','_03_09','_03_10','_04_00','_04_01','_04_02','_04_03','_04_04','_04_05','_04_06','_04_07','_05_04','_05_05','_05_06','_05_07','_05_13','_03_17','_03_18','_03_19','_03_20']
        train_types = ['_00_00','_00_01','_00_08','_00_09','_00_10','_00_11','_01_00','_01_01','_01_02','_01_03','_01_04','_01_05','_01_06','_01_07','_01_08','_02_00','_02_01','_02_02','_02_03','_02_04','_02_05','_02_06','_02_07','_02_08','_02_09','_03_00','_03_01','_03_02','_03_03','_03_04','_03_11','_03_12','_03_13','_03_14','_03_15','_03_16','_04_08','_04_09','_04_10','_04_11','_04_12','_05_00','_05_01','_05_02','_05_03','_05_08','_05_09','_05_10','_05_11','_05_12']


        all_imgs = open(root + phase + '.txt').readlines()
        imgs1 = []
        imgs = []
        for img in all_imgs:
            if '_06_' not in img:# and '_02_' not in img:
                imgs1.append(img)
        if 'train' in phase:
            self.types =  train_types
            for k in imgs1:
                ks = '_' + k.split('_')[1] + '_' + k.split('_')[2].split('.')[0]
                if ks in train_types:
                    imgs.append(k)
        else:
            self.types =  val_types
            for k in imgs1:
                ks = '_' + k.split('_')[1] + '_' + k.split('_')[2].split('.')[0]
                if ks in val_types:
                    imgs.append(k)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.load_bytes = load_bytes
        self.transform = transform
        all_target = {}
        for img in imgs:
            path, target = img.split('\n')[0].split(',')
            all_target[path] = np.float(target)
        self.all_target = all_target


    def __getitem__(self, index):
        path, target = self.imgs[index].split('\n')[0].split(',')
        target = np.float(target)
        name = '_' + path.split('_',1)[1].split('.')[0]
        ref_path = 'ref_' + path
        path1 = path.split('_')[0] + random.choice(self.types) + '.bmp'
        ref_path1 = path.split('_')[0] + '.bmp'
        target1 = self.all_target[path1]
        path = self.root + path
        path1 = self.root + path1
        ref_path = self.root + ref_path
        ref_path1 = self.root + ref_path1
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        ref_img = open(ref_path,'rb').read() if self.load_bytes else Image.open(ref_path).convert('RGB')
        img1 = open(path1,'rb').read() if self.load_bytes else Image.open(path1).convert('RGB')
        ref_img1 = open(ref_path1,'rb').read() if self.load_bytes else Image.open(ref_path1).convert('RGB')
        if self.transform is not None:
            img,ref_img = self.transform(img,ref_img)
            img1,ref_img1 = self.transform(img1,ref_img1)
        return np.concatenate((img,ref_img,img1,ref_img1),axis=0),  np.concatenate((np.array((target-900)/1000).reshape(1),np.array((target1-900)/1000).reshape(1)))
    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i].split(',')[0]) for i in indices]
            else:
                return [self.imgs[i].split(',')[0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x.split(',')[0]) for x in self.imgs]
            else:
                return [x.split(',')[0] for x in self.imgs]


