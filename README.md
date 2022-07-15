# ENetFall
EfficientNet based fall detection using channel state information (CSI). The project uses the Linux CSI tool (https://github.com/dhalperi/linux-80211n-csitool-supplementary) to collected CSI samples from host computers equiped with Intel NIC 5300 WiFi cards. The CSI samples are collected while several daily activities and falls are performed. Example CSI spectrograms are shown below for the activities involved.
![image](https://user-images.githubusercontent.com/8125847/179218645-60ae6d12-6265-466b-aefe-27185c53331e.png)

The CSI dataset includes 321 fall events and 436 non-fall events collected in 4 different indoor environments, performed by 22 volunteers. The dataset is used to train modified EfficientNet B0 and B1 to detect fall events. The diagram below shows the modification to the EfficientNet B0.
![image](https://user-images.githubusercontent.com/8125847/179219350-02554597-d9f0-4247-89da-0d413d5e46d4.png)

Use train.py to train the modified B0 / B1 with our dataset and use test.py to test the trained networks.

Required packages include pytorch, torchvision and scipy

Our CSI dataset and trained networks can be downloaded at
https://drive.google.com/file/d/1ehX8mjbZfNzXGzT7BrzHFB2F6ifoU9dv/view?usp=sharing


