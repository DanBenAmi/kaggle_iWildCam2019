
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import pickle

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import models
import torchvision.transforms as transforms
from datetime import date
from statistical_inference_project import WildDataset


test_df = pd.read_csv('input/test.csv')
test_dir = 'input/test_images'
print('Total images for test {0}'.format(len(os.listdir(test_dir))))
classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal',
                6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote',
                12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat',
                18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}


# Category distribution
class_hist = test_df['classes_wild'].value_counts()
print(class_hist)
plt.figure(figsize=(10, 5))
class_hist.plot(kind='bar', title="Category distribution", )
plt.show()
print(f"Only {len(class_hist)} classes are presented in the test set")


# reduce Class Indices
CLASSES_TO_USE = train_df_all['category_id'].unique()
NUM_CLASSES = len(CLASSES_TO_USE)
CLASSMAP = dict([(i, j) for i, j in zip(CLASSES_TO_USE, range(NUM_CLASSES))])
REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])
print(CLASSMAP)

aug = transforms.Compose([
    CLACHE(),
    # SimpleWhiteBalancing(),
    transforms.ToPILImage(),
    # transforms.Resize((747, 1024)),
    # transforms.Resize((100, 175)),
    transforms.Resize((60, 82)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_set = WildDataset(df=test_df,
                        img_dir=test_dir,
                        augs=aug)

# Data loader

test_loader = DataLoader(dataset=test_set, batch_size=50, shuffle=False)

class WildDataset(Dataset):
    def __init__(self, df, img_dir, augs=None):
        # self.df = df[:200]
        # self.df = df[df["file_name"].isin(os.listdir(r'/content/input/train_images'))]
        x = df[df["category_new_id"] == 1]
        x = x[:len(x) // 5]
        self.df = pd.concat([x, df[df["category_new_id"] != 1]])
        indices = np.arange(len(self.df))
        np.random.shuffle(indices)
        npdf = self.df.to_numpy()
        npdf = npdf[indices]
        self.df = pd.DataFrame(npdf, columns=self.df.columns)
        # self.df = self.df.iloc[:10000,:]
        self.img_dir = img_dir
        self.augs = augs
        self.sampler = self.make_sampler()

    def make_sampler(self):
        y_train = self.df['category_new_id']
        weights_mapper = {t: 1 / len(np.where(y_train == t)[0]) for t in np.unique(y_train)}
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





model = torch.load('ckpts/resnet50_trained_epoch-14_f1-0.8883_date-2023-02-23.pt')
with torch.no_grad():
    for batch in test_loader:
