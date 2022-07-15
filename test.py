import torch
import scipy.io as sio
import numpy as np
import torchvision
import torch.nn as nn
from train import to_device
from train import get_default_device, DeviceDataLoader
from train import evaluate
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as T


if __name__ == '__main__':
    device = get_default_device()
    num_classes = 2
    # load data, include one or multiple datasets
    # dataset_names = ['dataset_home_lab(L).mat', 'dataset_home_lab(R).mat', 'dataset_lecture_room.mat',
    #                  'dataset_living_room.mat', 'dataset_meeting_room.mat']
    dataset_names = ['dataset_meeting_room.mat','dataset_lecture_room.mat']
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
    max = torch.max(data_reshape)
    data_reshape = data_reshape / max
    mean = torch.mean(data_reshape)
    data_reshape = data_reshape - mean
    dataset = TensorDataset(data_reshape, labels_all)
    test_data_loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True, drop_last=True)
    test_data_loader = DeviceDataLoader(test_data_loader, device)
    testmodel = torchvision.models.efficientnet_b0(pretrained=True)
    in_features = testmodel.classifier[1].in_features
    # testmodel.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    testmodel.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    testmodel.load_state_dict(torch.load('ENetmod_pretrained.pth'))
    to_device(testmodel, device)
    num_incorrect_pred = 0
    testmodel.eval()
    result = evaluate(testmodel, test_data_loader)
    print(result)