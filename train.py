import os 
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import yaml
import dataloader
import argparse

parser = argparse.ArgumentParser(description='train maskrcnn.')
parser.add_argument("--config", help="Config file for Mask Rcn", default='./config.yaml')
args = parser.parse_args()
print(args.config)

        
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc)

    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
    
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


    
def main():

    with open(args.config, 'r') as f:
        yam = yaml.load(f)
    dataset_path = yam['dataset_path']
    batch_size = yam['batch_size']
    epochs = yam['epochs']
    gamma = yam['gamma']
    lr =yam['lr']
    momentum = yam['momentum']
    nc = yam['nc']
    num_workers = yam['num_workers'] 
    step_size = yam['step_size']
    weight_decay= yam['weight_decay']
    dataset = dataloader.Dataset(dataset_path)
    
    os.system('git clone https://github.com/pytorch/vision.git')
    os.system('cd vision')
    os.system('git checkout v0.3.0')
    
    os.system('cp references/detection/utils.py ../')
    os.system('cp references/detection/transforms.py ../')
    os.system('cp references/detection/coco_eval.py ../')
    os.system('cp references/detection/engine.py ../')
    os.system('cp references/detection/coco_utils.py ../')
    
    dataset = dataloader.Dataset(dataset_path, get_transform(train=True))
    dataset_test = dataloader.Dataset(dataset_path, get_transform(train=False))
    
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    
    data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)    

    model = get_instance_segmentation_model(nc)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader,  device,epoch, print_freq=10)
        lr_scheduler.step()
        print(evaluate(model, data_loader_test, device=device))     
    
    if __name__ == "__main__":
        main()
