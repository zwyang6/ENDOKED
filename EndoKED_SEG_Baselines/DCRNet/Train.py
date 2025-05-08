import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.loss import BCEDiceLoss
from utils.dataloader import get_loader
from utils.dataloader import test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
from lib.DCRNet import DCRNet
from tqdm import tqdm
from utils.metrics import Metrics
from utils.metrics import evaluate
import logging
import numpy as np
import random 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import warnings
warnings.filterwarnings('ignore')

def valid(model, dataset, opt):

    opt.valid_data_dir = f'{opt.val_root}/{dataset}/'
    image_root = '{}/images/'.format(opt.valid_data_dir)
    gt_root = '{}/masks/'.format(opt.valid_data_dir)
    valid_dataloader = test_dataset(image_root, gt_root, opt.testsize)
    total_batch = int(len(os.listdir(gt_root)) / 1)
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    bar = tqdm(valid_dataloader.images)  
    for i in bar:
        image, gt, name = valid_dataloader.load_data()
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = gt.cuda()
        
        pred = model(image)
        output = pred[len(pred)-1].sigmoid()

        _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

        metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)

    metrics_result = metrics.mean(total_batch)

    return metrics_result, total_batch


def train(train_loader, model, optimizer):
    model.train()
    best = 0
    best_idx = 0
    # ---- multi-scale training ----
    for epoch in range(opt.epoch):
        loss_record = AvgMeter()
        for i, pack in enumerate(train_loader, start=1):           
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack['image'], pack['label']
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- forward ----
            pred = model(images)
            # ---- loss function ----
            loss4 = BCEDiceLoss(pred[4], gts)
            loss3 = BCEDiceLoss(pred[3], gts)
            loss2 = BCEDiceLoss(pred[2], gts)
            loss1 = BCEDiceLoss(pred[1], gts)
            loss0 = BCEDiceLoss(pred[0], gts)
            loss = loss0 + loss1 + loss2 + loss3 + loss4
            # ---- backward ----
            loss.backward()
            optimizer.step()
            # ---- recording loss ----
            loss_record.update(loss.data, opt.batchsize)
            # ---- train visualization ----
            if opt.local_rank == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[loss: {:.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record.show()))


        if opt.local_rank == 0:
            logging.info(f"####################################Testing_EPOCH{epoch}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        total_dice = 0.0
        total_images = 0
        # for dataset in ['kvasir-Selected_0.909','CVC-ClinicDB-Selected', 'CVC-ColonDB', 'CVC-300',  'ETIS-LaribPolypDB', 'polygen']:
        # for dataset in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen']:
        for dataset in [ 'Kvasir','CVC-ClinicDB']:

            metrics_result, num_images = valid(model, dataset,opt)
            total_dice += metrics_result['F1'] * num_images
            total_images += num_images
            if opt.local_rank == 0:
                print(f'TrainingEpoch[{epoch}]:\tTested on Dataset:[{dataset}]>>>>>>>>>')
                print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))
                mdice,miou,mprecision,mrecall=metrics_result['F1'], metrics_result['IoU_poly'], metrics_result['precision'],metrics_result['recall'],
                logging.info(f'TrainingEpoch[{epoch}]_Tesed_on_[{dataset}]:\tDice:{mdice}\tIoU:{miou}\tPrecision:{mprecision}\tRecall:{mrecall}')  

        if opt.local_rank == 0:
            meandice = total_dice/total_images
            if meandice > best:
                best_idx = epoch
                best = meandice
                torch.save(model.state_dict(), f'{opt.record_path}/{opt.dataset}_{epoch}_Best.pt')
            print(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
            logging.info(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))

    if opt.local_rank == 0:
        torch.save(model.state_dict(), f'{opt.record_path}/{opt.dataset}_Last.pt')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=512, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=224, help='training dataset size')
    parser.add_argument('--testsize', type=int,
                        default=224, help='training dataset size')
    parser.add_argument('--train_path', type=str,
                        default='/data2/yinzijin/dataset/EndoScene/TrainDataset/', help='path to train dataset')
    parser.add_argument('--save_root', type=str,
                        default='DCRNet_EndoScene')
    parser.add_argument('--gpu', type=str,
                        default='0', help='GPUs used')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='Train_on_KvasirandDB')
    parser.add_argument('--val_root', type=str, default='/data2/zhangruifei/polypseg')

    parser.add_argument('--tag', type=str,
                        default='scratch', help='decay rate of learning rate')
    parser.add_argument('--cpt_path', type=str,
                        default='./pretrained.pth', help='decay rate of learning rate')
    parser.add_argument('--load_pretrained',
                        default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")

    opt = parser.parse_args()

    # ---- build models ----
    setup_seed(opt.seed)

    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl', )

    if opt.local_rank == 0:
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        opt.record_path = f'./logs/{opt.dataset}/{timestamp}_{opt.tag}/'
        os.makedirs(opt.record_path,exist_ok=True)
        opt.log_path = opt.record_path + 'train_log.log'
        logging.basicConfig(filename=opt.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = DCRNet()
    model.cuda()

    if opt.load_pretrained:
        print("LOADING CPT from ENDOKED")
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LOADING CPT from ENDOKED")
        cpt = torch.load(opt.cpt_path,map_location='cpu')
        new = {}
        for k_,v in cpt.items():
            if 'module.' in k_:
                k = k_.replace('module.','')
                new[k] = v
        model.load_state_dict(new)

    model = DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(opt, image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    if opt.local_rank == 0:
        print("#"*20, "Start Training", "#"*20)
    
    train(train_loader, model, optimizer)