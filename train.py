import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import Logger
from tqdm import tqdm
from dataloaders.pic_data import ImgDataset
from dataloaders.CVDcifar import CVDcifar
from network import ViT
from utils.cvdObserver import cvdSimulateNet
from utils.conditionP import conditionP

# hugface官方实现
# from transformers import ViTImageProcessor, ViTForImageClassification
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits

dataset = 'local'
num_classes = 6

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=4)
parser.add_argument('--size',type=int, default=32)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()

### write model configs here
save_root = './run'
pth_location = './Models/model_new.pth'
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
train_val_percent = 0.8
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# skf = StratifiedGroupKFold(n_splits=n_splits)

trainset = CVDcifar('./',train=True,download=True,patch_size=args.patch)
testset = CVDcifar('./',train=False,download=True,patch_size=args.patch)
inferenceset = CIFAR10('./',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),]))

train_size = int(len(trainset) * train_val_percent)
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True)
testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)
inferenceloader = torch.utils.data.DataLoader(inferenceset,batch_size=args.batchsize,shuffle = False,)
# trainval_loader = {'train' : trainloader, 'valid' : validloader}

model = ViT('ColorViT', pretrained=False,image_size=32,patches=4,num_classes=4*4*3)
model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[7,20],gamma=0.3)
logger.auto_backup('./')

def train(trainloader, model, criterion, optimizer, lrsch, logger, args, epoch):
    model.train()
    loss_logger = 0.
    logger.update_step()
    for img, ci_patch, ci_rgb in tqdm(trainloader,ascii=True,ncols=60):
        optimizer.zero_grad()

        outs = model(img.cuda(),ci_patch.cuda())   
        # print("opt tensor:",out)
        ci_rgb = ci_rgb.cuda()

        # if epoch>30:
        #     # 冻结部分层
        #     for name, param in model.named_parameters():
        #         if ("transformer" in name):
        #             param.requires_grad = False
        loss_batch = criterion(outs,ci_rgb)
        loss_batch.backward()
        loss_logger += loss_batch.item()    # 显示全部loss
        optimizer.step()
        lrsch.step()

    loss_logger /= len(trainloader)
    print("Train loss:",loss_logger)
    log_metric('Train', logger,loss_logger)
    if not (logger.global_step % args.save_interval):
        logger.save(model,optimizer, lrsch, criterion)
        
def validate(testloader, model, criterion, optimizer, lrsch, logger, args):
    model.eval()
    loss_logger = 0.

    for img, ci_patch, ci_rgb in tqdm(testloader,ascii=True,ncols=60):
        with torch.no_grad():
            outs = model(img.cuda(),ci_patch.cuda())   
        ci_rgb = ci_rgb.cuda()
        # print("label:",label)
        
        loss_batch = criterion(outs,ci_rgb)
        loss_logger += loss_batch.item()    # 显示全部loss

    loss_logger /= len(testloader)
    print("Val loss:",loss_logger)

    acc = log_metric('Val', logger,loss_logger)

    return acc, model.state_dict()

def sample_enhancement(model,inferenceloader,epoch):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    其中条件概率假定服从N(f(I^{cvd},c_i^{cvd}),I) f为重新上色学习器
    '''
    model.eval()
    cvd_process = cvdSimulateNet(cuda=True,batched_input=True) # 保证在同一个设备上进行全部运算
    for img,_ in inferenceloader:
        img_cvd = cvd_process(img.cuda())
        img_cvd:torch.Tensor = img_cvd[0,...].unsqueeze(0)  # shape 1,C,H,W
        img_t:torch.Tensor = img[0,...].unsqueeze(0)
        break   # 只要第一张
    img_out = img_t.clone()
    inference_criterion = conditionP()

    for i in tqdm(range(args.size//args.patch)):
        for j in range(args.size//args.patch):
            img_t_patch = img_t[:,:,i*args.patch:(i+1)*args.patch,j*args.patch:(j+1)*args.patch].clone().cuda()    # 重新调色后的patch
            img_t_patch.requires_grad = True
            # img_cvd_patch = cvd_process(img_t_patch).cuda()
            img_ori_patch = img_t[:,:,i*args.patch:(i+1)*args.patch,j*args.patch:(j+1)*args.patch]  # 作为GT的patch
            inference_optimizer = torch.optim.SGD(img_t_patch,lr=args.lr,momentum=0.3)   # 对输入图像进行梯度下降
            for iter in range(50):
                inference_optimizer.zero_grad()
                img_cvd_patch = cvd_process(img_t_patch).cuda()
                out = model(img_cvd,img_cvd_patch)
                loss = inference_criterion(out,img_ori_patch.cuda())    # 相当于-log p(img_ori_patch|img_cvd,img_t_patch)
                loss.backward()
                inference_optimizer.step()

            img_out[:,:,i*args.patch:(i+1)*args.patch,j*args.patch:(j+1)*args.patch] = img_t_patch
    img_out_array = img_out.squeeze(0).cpu().detach().numpy()
    plt.imshow(img_out_array)
    plt.savefig('./run/'+f'sample_e{epoch}.png')


def log_metric(prefix, logger, loss):
    logger.log_scalar(prefix+'/loss',loss,print=False)
    return 1/loss   # 越大越好

testing = validate
auc = 0

if args.test == True:
    finaltestset = CVDcifar('./',train=False,download=True)
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = False,num_workers=8)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    testing(finaltestloader,model,criterion,optimizer,lrsch,logger,args)
else:
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        train(trainloader, model,criterion,optimizer,lrsch,logger,args,i)
        score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args)
        sample_enhancement(model,inferenceloader,i)
        if score > auc:
            auc = score
            torch.save(model_save, pth_location)
