import torch
import torch.nn as nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from tqdm import tqdm
from augmentations import IsotropicResize
import cv2

def create_train_transform(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.6),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )


def naive_train_epoch(current_epoch, model, optimizer, loss_function, train_data_loader):

    print("training epoch {}".format(current_epoch))

    model.train()
    running_loss = 0

    for i, sample in tqdm(enumerate(train_data_loader)):
        img = sample["image"]
        labels = sample["labels"]
        out_labels = model(img)

        # print(out_labels)
        # print(labels)

        loss = loss_function(out_labels, torch.tensor(labels).float())
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return running_loss



def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
        local_rank, only_valid):
    
    losses = AverageMeter()
    fake_losses = AverageMeter()
    real_losses = AverageMeter()
    max_iters = 4
    print("training epoch {}".format(current_epoch))

    model.train()

    for i, sample in tqdm(enumerate(train_data_loader)):
        imgs = sample["image"]
        labels = sample["labels"]
        out_labels = model(img)
        if only_valid:
            valid_idx = sample["valid"].float() > 0
            out_labels = out_labels[valid_idx]
            labels = labels[valid_idx]
            if labels.size(0)==0:
                continue

        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5

        if torch.sum(fake_idx * 1) > 0:
            fake_loss = loss_functions["classifier_loss"](out_labels[fake_idx],labels[fake_idx])

        if torch.sum(real_idx * 1) > 0:
            real_loss = loss_functions["classifier_loss"](out_labels[real_idx],labels[real_idx])


        loss = (fake_loss + real_loss) / 2
        losses.update(loss.item(), imgs.size(0))
        fake_losses.update(0 if fake_loss == 0 else fake_loss.item(), imgs.size(0))
        real_losses.update(0 if real_loss == 0 else real_loss.item(), imgs.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if i == max_iters - 1:
            break


class NaiveClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = Linear(3*300*300,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x.flatten())
        x = self.sigmoid(x)
        return x


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate = 0.0):
        super().__init__()
        self.encoder = encoder
        self.avg_pool = AdaptiveAvgPool2d((1,1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(100, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_epoch(model, dataset, criterion, optimizer):

    features = dataset["features"]
    labels = dataset["labels"]

    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)

    features = features.float()
    labels = labels.long()

    optimizer.zero_grad()

    outputs = model(features)
    loss = criterion(outputs, labels)
    preds = outputs >= 0.5

    loss.backward()
    optimizer.step()

    accuracy = torch.sum(preds)/preds.shape[0]

    return loss.item(), accuracy 