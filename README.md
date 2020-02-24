# pytorch-semantic-segmentation
semantic segmentation for carvana dataset

Goal:
Find mask for car image

Use:
1. Pytorch
2. Pretrained torchvision model (deeplab v3)
3. Feature extraction for models classifier
4. Custom dataset subclass
5. Custom dataloader
6. Hand-writing train and eval func
7. Helper funcs for image show

Used dataset:
Just train images and masks from "Carvana" dataset
https://www.kaggle.com/c/carvana-image-masking-challenge/data

Folder struct:
--dataset\
---train_data\
----image1
----imageN
---train_masks\
----mask1
----maskN
