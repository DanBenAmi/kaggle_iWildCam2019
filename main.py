#!/usr/bin/env python
# coding: utf-8


from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from transform import CLACHE, SimpleWhiteBalancing, WhiteBalancing, WhiteBalancing2

from utills import *

import os
import torch
import torchvision

from torch.utils.data import WeightedRandomSampler
import torchvision.models
import torch.nn as nn
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Custom data generator
class WildDataset(Dataset):
    def __init__(self, df, img_dir, augs=None):


        # self.df = df[df["file_name"].isin(os.listdir(r'/Users/elad.sofer/src/kaggle_iWildCam2019/input/train_test'))]
        self.df = df
        self.img_dir = img_dir
        self.augs = augs
        self.sampler = self.make_sampler()

    def make_sampler(self):
        y_train = self.df['category_new_id']
        weights_mapper = {t: 1/len(np.where(y_train == t)[0]) for t in np.unique(y_train)}
        # weight = 1 / class_sample_count
        sample_weights = np.array([weights_mapper[t] for t in y_train])
        sample_weights = torch.from_numpy(sample_weights)
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(self.df), replacement=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        label = self.df.iloc[idx, 1]
        if self.augs is not None:
            image = self.augs(image)
        return image, label


def show_aug(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


# ### Data Evaluation

# Data path
train_df_all = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
# sub = pd.read_csv('input/sample_submission.csv')
train_dir = 'input/train_images'
test_dir = 'input/test_images'
print('Total images for train {0}'.format(len(os.listdir(train_dir))))
# print('Total images for train {0}'.format(len(os.listdir(test_dir))))

# train_df_all.iloc[50:60]

# test_df.iloc[50:60]


# code from https://www.kaggle.com/gpreda/iwildcam-2019-eda
classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal',
                6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote',
                12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat',
                18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}

train_df_all['classes_wild'] = train_df_all['category_id'].apply(lambda cw: classes_wild[cw])
# train_df_all.iloc[50:60]

# Category distribution
class_hist = train_df_all['classes_wild'].value_counts()
print(class_hist)
plt.figure(figsize=(10, 5))
class_hist.plot(kind='bar', title="Category distribution", )
plt.show()
print(f"Only {len(class_hist)} classes are presented in the train set")

# As seen in the histogram we are witnessing here for a very imbalanced data, as the 'empty' class is far bigger than all other classes combined.
# Furthermore, even when ignoring the 'empty' class, the data is still very imbalanced between the classes And some classes are not even train set at all, i.e. only 14 out of 23 classes are presented in the train set.

# ### Reduce Class Indices

CLASSES_TO_USE = train_df_all['category_id'].unique()
NUM_CLASSES = len(CLASSES_TO_USE)
CLASSMAP = dict([(i, j) for i, j in zip(CLASSES_TO_USE, range(NUM_CLASSES))])
REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])
print(CLASSMAP)
# define new id
train_df_all['category_new_id'] = train_df_all['category_id'].map(CLASSMAP)
train_df = train_df_all[['file_name', 'category_new_id']]

# ### Data Loading

# train_df

train, val = train_test_split(train_df, stratify=train_df.category_new_id, test_size=0.1)

aug = transforms.Compose([
    CLACHE(),
    SimpleWhiteBalancing(),
    transforms.ToPILImage(),
    # transforms.Resize((747, 1024)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# iWildCam dataset
train_set = WildDataset(df=train,
                        img_dir=train_dir,
                        augs=aug)

valid_set = WildDataset(df=val,
                        img_dir=train_dir,
                        augs=aug)

# Data loader

train_loader = DataLoader(dataset=train_set, batch_size=24, sampler=train_set.sampler)
val_loader = DataLoader(dataset=valid_set, batch_size=24, shuffle=False)

# show_aug(train_set[544][0])
#
# img = train_set[789][0].numpy()
#
# # Using cv2.imshow() method
# # Displaying the image
# plt.imshow(img.transpose((1, 2, 0)))
# plt.show()
# pass






num_epochs = 20
lr = 5.5e-5


model_fixed = ResNet(base_model_name='resnet50',num_classes=len(CLASSES_TO_USE))
# model_fixed = to_device(two_CNN(), device)
optimizer = torch.optim.Adam(model_fixed.parameters(), lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

history_ResNet_FT = train_model(name='model_ft', epochs=num_epochs,
                                model=model_fixed, train_loader=train_loader,
                                val_loader=train_loader,
                                optimizer=optimizer, scheduler=exp_lr_scheduler)

torch.save(history_ResNet_FT,'/content/drive/MyDrive/DNN final/histories/history_resnet_ft.pt')