import torch
import math
import numpy as np
import torch.nn as nn
import scipy.io as sio
import torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as T


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to the device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_default_device():
    """"Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensors to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def precision(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    num_tp = 0
    num_fp = 0
    for idx in range(len(labels)):
        if preds[idx].item() == labels[idx].item() and preds[idx].item() == 1:
            num_tp = num_tp + 1
        if preds[idx].item() != labels[idx].item() and preds[idx].item() == 1:
            num_fp = num_fp + 1
    return torch.tensor(num_tp / (num_tp + num_fp))


def recall(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    num_tp = 0
    num_fn = 0
    for idx in range(len(labels)):
        if preds[idx].item() == labels[idx].item() and preds[idx].item() == 1:
            num_tp = num_tp + 1
        if preds[idx].item() != labels[idx].item() and preds[idx].item() == 0:
            num_fn = num_fn + 1
    return torch.tensor(num_tp / (num_tp + num_fn))


def validation_step(model, batch):
    images, labels = batch
    out = model(images)
    labels = labels.squeeze()
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    prec = precision(out, labels)
    rec = recall(out, labels)
    return {'val_loss': loss.detach(), 'val_acc': acc, 'val_prec': prec, 'val_rec': rec}


def epoch_end(model, epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:4f}"
            .format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def validation_epoch_end(model, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    batch_precs = [x['val_prec'] for x in outputs]
    epoch_prec = torch.stack(batch_precs).mean()
    batch_recs = [x['val_rec'] for x in outputs]
    epoch_rec = torch.stack(batch_recs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_prec': epoch_prec.item(), 'val_rec': epoch_rec.item()}


def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def training_step(model, batch):
    images, labels = batch
    out = model(images)
    labels = labels.squeeze()
    loss = F.cross_entropy(out, labels)
    return loss


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, desired_acc,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, ):
    global stop
    torch.cuda.empty_cache()
    history = []
    # set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # set up one-cycle lr scheduler
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss)
            loss.backward()
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record and update lr
            lrs.append(get_lr(optimizer))
            sched.step()

        images, labels = batch
        out = model(images)
        labels = labels.squeeze()
        acc = accuracy(out, labels)
        print("Train acc is ", acc)

        # Validation
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        if result['val_acc'] >= desired_acc and epoch > 40:
            stop = False
            torch.save(model_ft.state_dict(), 'ENetmod_pretrained.pth')
            return history
    return history


if __name__ == '__main__':
    global stop
    stop = True
    num_classes = 2  # fall and non fall
    batch_size = 60
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-4
    device = get_default_device()
    # load data, include one or multiple datasets
    # dataset_names = ['dataset_home_lab(L).mat', 'dataset_home_lab(R).mat', 'dataset_lecture_room.mat', 'dataset_living_room.mat'
    #                ,'dataset_meeting_room.mat']
    dataset_names = [ 'dataset_meeting_room.mat', 'dataset_lecture_room.mat']


    for idx in range(len(dataset_names)):
        data = sio.loadmat(dataset_names[idx])
        if 'data_all' in locals():
            temp_d = data['dataset_CSI_t']  # 3-d CSI data, num_items * num_CSI_timesamples * num_subcarrier
            temp_l = data['dataset_labels']
            data_all = np.concatenate((data_all, temp_d), axis=0)
            labels_all = np.concatenate((labels_all, temp_l), axis=None)
        else:
            data_all = data['dataset_CSI_t']
            labels_all = data['dataset_labels']

    # prepare data, nSample x nChannel x width x height
    # reshape train data size to nSample x nSubcarrier x 1 x 1
    num_train_instances = data_all.shape[0]
    data_all_3ch = np.ndarray(shape=(num_train_instances, 3, 625, 30))
    # ch1 = data from RX1, ch2 from RX2, ch3 from RX3
    data_all_3ch[:, 0, :, :] = data_all[:, :, 0:90:3]
    data_all_3ch[:, 1, :, :] = data_all[:, :, 1:90:3]
    data_all_3ch[:, 2, :, :] = data_all[:, :, 2:90:3]
    data_reshape = torch.from_numpy(data_all_3ch).type(torch.FloatTensor).view(num_train_instances, 3, 625, 30)
    # dataset transform
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for idx in range(num_train_instances):
        data_reshape[idx] = transform(data_reshape[idx])

    labels_all = torch.from_numpy(labels_all).type(torch.LongTensor)
    indices = torch.randperm(len(data_reshape)).tolist()
    testset_size = math.ceil(num_train_instances * 0.2)
    max = torch.max(data_reshape[indices[:-testset_size]])
    data_reshape = data_reshape / max
    mean = torch.mean(data_reshape[indices[:-testset_size]])
    data_reshape = data_reshape - mean

    dataset = TensorDataset(data_reshape, labels_all)
    # split data set into train and test
    train_dataset = torch.utils.data.Subset(dataset, indices[:-testset_size])  # everything except the last X

    test_dataset = torch.utils.data.Subset(dataset, indices[-testset_size:])  # the last X
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_data_loader = DeviceDataLoader(train_data_loader, device)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_data_loader = DeviceDataLoader(test_data_loader, device)

    while stop:
        # Get pretrained model
        model_ft = torchvision.models.efficientnet_b0(pretrained=True)
        in_features = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        history = []
        desired_acc = 0.94
        history += fit_one_cycle(num_epochs, learning_rate, model_ft, train_data_loader, test_data_loader, desired_acc, grad_clip=None,
                             weight_decay=weight_decay, opt_func=torch.optim.Adam)
