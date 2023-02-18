
import torch.nn.functional as F

import sklearn


import torch

from torchvision import models
import torch.nn as nn

import numpy as np



def calc_metric(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    f1 = sklearn.metrics.f1_score(labels, preds, average='macro')
    return acc.item(), f1


# @torch.no_grad()
def evaluate(model, name, vl_loader):
    """
    Evaluate a model upon a validation DataLoader, the function calculates the
    accuracy and loss for each architecture depends on the mode.
    in mode='encoder', an autoencoder object must be supplied.
    """
    losses_lst, acc_lst, f1_lst = np.array([]), np.array([]), np.array([])
    # model.eval()
    with torch.no_grad():
        outputs = []
        for batch in vl_loader:
            images, labels = batch
            # Evaluation upon a NN architecture
            out = model.forward(images)
            acc, f1 = calc_metric(out, labels)
            loss = F.cross_entropy(out, labels)

            losses_lst = np.append(losses_lst, loss.item())
            acc_lst = np.append(acc_lst, acc)
            f1_lst = np.append(f1_lst, f1)

    return {'{}_loss'.format(name): losses_lst.mean(),
            '{}_acc'.format(name): acc_lst.mean(),
            '{}_f1'.format(name): f1_lst.mean()}


def train_model(name, epochs, model, train_loader, val_loader, optimizer, scheduler=None):
    """
    Train a model upon a train DataLoader, the function calculates the
    accuracy and loss for each architecture depends on the mode.
    in mode='encoder', an autoencoder object must be supplied.
    """
    history = []
    best_acc = 0
    min_loss = 100
    loss_func = F.cross_entropy

    for epoch in range(epochs):
        model.train()
        # Training Phase
        train_losses = []
        train_acc = []
        train_f1 = []

        for batch in train_loader:
            images, labels = batch

            # Cleanup
            optimizer.zero_grad()

            # Train a classic NN Model
            outputs = model.forward(images)
            train_loss = loss_func(outputs, labels)

            tr_acc, tr_f1 = calc_metric(outputs, labels)
            # History Tracking
            train_acc.append(tr_acc)
            train_losses.append(train_loss)
            train_f1.append(tr_f1)
            # Backprop & update weights
            train_loss.backward()
            optimizer.step()

        # Learning rate decay
        if scheduler:
            scheduler.step()

        # Finished an epoch, calculate the validation accuracy
        result = evaluate(model, 'val', val_loader)  # results holds both train and val
        result['train_loss'] = torch.stack(train_losses).mean().item()
        temp_acc = float(result['val_acc'])
        temp_loss = float(result['val_loss'])
        result['train_acc'] = np.array(train_acc).mean().item()

        print("Epoch {}: train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f} ".format(
            epoch + 1, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'], result['val_f1']))

        # if temp_acc > best_acc:
        #     best_acc = temp_acc
        #     torch.save(model,'/content/drive/MyDrive/DNN final/checkpoints/{}_trained_epoch{}_acc{:.4f}.pt'.format(name,epoch,best_acc))
        history.append(result)

    return history


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# class ResNet(nn.Module):
#     def __init__(self, tl_mode='fixed', n=512, classes_num=14):
#         super().__init__()
#         # Use a pretrained model
#         self.network = torchvision.models.resnet50(pretrained=True)
#         num_ftrs = self.network.fc.in_features
#         # ConvNet as fixed feature extractor mode
#         if tl_mode == 'fixed':
#             # Fixing the Resnet-50 Layers
#             for param in self.network.parameters():
#                 param.requires_grad = False
#
#             trail_nn = nn.Sequential(nn.Linear(num_ftrs, classes_num))
#                 # ,
#                 # nn.Linear(n, 64), nn.ReLU(),
#                 # nn.Linear(64, 16), nn.ReLU(),
#                 # nn.Linear(16, classes_num))
#         else:
#             # Fine-tuning ConvNet mode
#             trail_nn = nn.Linear(num_ftrs, classes_num)
#
#         # Replace last layer
#         self.network.fc = trail_nn
#
#     def forward(self, xb):
#         return torch.sigmoid(self.network(xb))

class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=True,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        x = self.classifier(x)
        # multiclass_proba = F.softmax(x, dim=1)
        # multilabel_proba = F.sigmoid(x)
        # return {
        #     "logits": x,
        #     "multiclass_proba": multiclass_proba,
        #     "multilabel_proba": multilabel_proba
        # }
        return x


class two_CNN(nn.Module):
    '''This model have two CNN layes and two fully connected layes'''

    def __init__(self, n=100):
        super(two_CNN, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=5, padding=int((5 - 1) / 2))
        self.conv2 = nn.Conv2d(self.n, 2 * self.n, kernel_size=5, padding=int((5 - 1) / 2))
        self.fc1 = nn.Linear(7200 * 24, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=14)
        self.name = "two_CNN"

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=5, stride=1, padding=1)(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)

        x = torch.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
