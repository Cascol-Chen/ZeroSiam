from logging import debug
import os
import time
import argparse
import json
import random
import numpy as np
from pycm import *

import math
from typing import ValuesView

from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.create_incorrect_labels_datasets import create_error_grouped_datasets
from utils.cli_utils import *

import torch    
import torch.nn.functional as F

from tta_library import tent, eata, sar, deyo, deyo_come, zerosiam
from tta_library.sam import SAM
import timm
from models.byol_wrapper import BYOLWrapper


import models.Res as Resnet
from utils.metrics import ECELoss
import warnings
warnings.filterwarnings("ignore")
# os.environ['TORCH_HOME'] = '/chenyaofo/cgh/cowork/cdy/research/checkpoints'

import matplotlib.pyplot as plt


def get_energy(x):
    return -torch.log(torch.exp(x).sum(1))

def get_entropy_score(logits: torch.Tensor, eps=1e-6) -> torch.Tensor:
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1) / math.log(logits.size(-1))

def adapt_only(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    energy = AverageMeter('Energy', ':6.2f')
    confidence = AverageMeter('Confidence', ':6.3f')
    normVal = AverageMeter('Norm', ':6.2f')
    Enp = AverageMeter('Enp', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, energy, confidence, normVal, Enp],
        prefix='Test: ')
    model.eval()

    outputs_list, targets_list = [], []
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()

            output = model(images)    
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())
            # _, targets = output.max(1)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            ene = get_energy(output)
            conf = output.softmax(1).max(1)[0]
            norm_val = torch.norm(output, p=2, dim=1)

            top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))
            energy.update(ene.mean(), images.size(0))
            confidence.update(conf.mean(), images.size(0))
            normVal.update(norm_val.mean(), images.size(0))
            Enp.update(get_entropy_score(output).mean(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % (args.print_freq) == 0:
                progress.display(i)
        args.print_freq = 50000 // 20 // 64
    return model

def validate(val_loader, model, criterion, args, mode='eval'):
    if args.exp_type in ["incorrect_labels_k1", "incorrect_labels_k5"]: 
        model = adapt_only(args.error_val_loader, model, None, args, mode='eval')
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    energy = AverageMeter('Energy', ':6.2f')
    confidence = AverageMeter('Confidence', ':6.3f')
    normVal = AverageMeter('Norm', ':6.2f')
    Enp = AverageMeter('Enp', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, energy, confidence, normVal, Enp],
        prefix='Test: ')
    model.eval()

    outputs_list, targets_list = [], []
    with torch.no_grad():
        end = time.time()
        for j in range(1):
            # top1 = AverageMeter('Acc@1', ':6.2f')
            # top1.reset()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                    # print(f'images: {images}')
                if torch.cuda.is_available():
                    target = target.cuda()
                    # print(f'target: {target}')

                if args.exp_type in ["incorrect_labels_k1", "incorrect_labels_k5"]: # 只评估
                    if args.method == "zerosiam":
                        output = model.model(images)[0]
                    elif args.method == "no_adapt":
                        output = model(images)
                    else:
                        output = model.model(images)
                else:
                    # torch.cuda.reset_peak_memory_stats()
                    output = model(images) 
                    # print(f'memory usage: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
                outputs_list.append(output.cpu())
                targets_list.append(target.cpu())
                # _, targets = output.max(1)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                ene = get_energy(output)
                conf = output.softmax(1).max(1)[0]
                norm_val = torch.norm(output, p=2, dim=1)

                # print(acc1[0])
                top1.update(acc1[0], images.size(0))
                # top5.update(acc5[0], images.size(0))
                energy.update(ene.mean(), images.size(0))
                confidence.update(conf.mean(), images.size(0))
                normVal.update(norm_val.mean(), images.size(0))
                Enp.update(get_entropy_score(output).mean(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % (args.print_freq) == 0:
                    progress.display(i)
                    # progress.display(i)
                    # i+=1
            # if j == 2:
            #     top1_3rd_epoch = top1.avg
            # if j == 4:
            #     top1_5th_epoch = top1.avg
        outputs_list = torch.cat(outputs_list, dim=0).numpy()
        targets_list = torch.cat(targets_list, dim=0).numpy()
        ece_avg = ECELoss().loss(outputs_list, targets_list) * 100
    # return top1_3rd_epoch, top1_5th_epoch, ece_avg
    return top1.avg, top1.avg, ece_avg

def get_args():

    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--data_type', default='imagenet-c', type=str, help='imagenet-c, sketch')
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=64, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    
    # Exp Settings
    parser.add_argument('--method', default='sar', type=str, help='no_adapt, tent, eata, sar')
    parser.add_argument('--model', default='vitbase_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='label_shifts', type=str, help='normal, mix_shifts, bs1, label_shifts')
    parser.add_argument('--exp_setting', default='each_shift_reset', type=str, help='each_shift_reset, continual')
    parser.add_argument('--lr_scale', type=float, default=1, help='\epsilon in Eqn. (5) for filtering redundant samples')
    parser.add_argument('--lr_p', type=float, default=1, help='\epsilon in Eqn. (5) for filtering redundant samples')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;')

    # tag
    parser.add_argument('--tag1', default='', type=str, help='experiment tag')
    parser.add_argument('--tag2', default='', type=str, help='detailed tag')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True


    if not os.path.exists(args.output): # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)

    tag1_folder = os.path.join(args.output, args.tag1)
    if not os.path.exists(tag1_folder):
        os.makedirs(tag1_folder, exist_ok=True)
    print(tag1_folder)

    tag2_folder = os.path.join(tag1_folder, args.tag2)
    if not os.path.exists(tag2_folder):
        os.makedirs(tag2_folder, exist_ok=True)
    print(tag2_folder)

    args.logger_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-{}-{}-{}-{}-seed{}-{}-{}-level{}.txt".format(
                                    args.method, args.data_type, args.exp_type, 
                                    args.exp_setting, args.seed, args.tag2, args.model, args.level)
    logger = get_logger(name="project", output_directory=tag2_folder, log_name=args.logger_name, debug=False)
    
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    
    # common_corruptions = ['gaussian_noise']
    

    if args.exp_type == 'mix_shifts':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)

            val_dataset, _ = prepare_test_data(args)
            if args.method in ['tent', 'eata', 'sar', 'no_adapt','deyo','deyo_come', 'zerosiam']:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            datasets.append(val_dataset)

        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.test_batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)
        common_corruptions = ['mix_shifts']
    
    if args.exp_type == 'bs1' or args.exp_type == 'label_shifts+bs1':
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")
    

    if args.exp_type == 'label_shifts' or args.exp_type == 'label_shifts+bs1':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")


    acc1s, acc5s = [], []
    ece_list = []
    ir = args.imbalance_ratio
    logger.info(args)
    for corrupt in common_corruptions:
        args.corruption = corrupt
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs
        # args.print_freq = 1
        

        if args.method in ['tent', 'eata', 'sar', 'no_adapt','deyo','deyo_come', 'zerosiam']:
            if args.corruption != 'mix_shifts':
                if args.exp_type == 'bs1' or args.exp_type == 'label_shifts+bs1' :
                    args.test_batch_size = 1
                # elif args.exp_type == 'label_shifts': # ls+bs1 10w samples setting
                #     args.test_batch_size = 1
                else:
                    args.test_batch_size = 64
                val_dataset, val_loader = prepare_test_data(args)
                val_dataset.switch_mode(True, False)
                if args.exp_type in ["incorrect_labels_k1", "incorrect_labels_k5"]: # blind-spot setting
                    # -------------incorrect + bs1:-------------
                    args.test_batch_size = 1
                    bs = args.test_batch_size
                    args.print_freq = 50000 // 20 // bs
                    # -------------incorrect + bs1:-------------
                    args.if_shuffle = False # 用来adapt的shuffle关掉
                    error_val_dataset, error_val_loader = prepare_test_data(args)
                    error_val_dataset.switch_mode(True, False)
                    args.if_shuffle = True # 用来评估incorrect TTA的shuffle要开
        else:
            assert False, NotImplementedError
        # construt new dataset with online imbalanced label distribution shifts, see Section 4.3 for details
        # note that this operation does not support mix-domain-shifts exps
        if args.exp_type == 'label_shifts' or args.exp_type == 'label_shifts+bs1':
            logger.info(f"imbalance ratio is {ir}")
            if args.seed == 2021:
                indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, ir)
            else:
                indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed, 100000, ir)
            logger.info(f"label_shifts_indices_path is {indices_path}")
            indices = np.load(indices_path)
            val_dataset.set_specific_subset(indices.astype(int).tolist())
        
        
        # build model for adaptation
        if args.method in ['tent', 'eata', 'sar', 'no_adapt','deyo','deyo_come', 'zerosiam']:
            if args.model == "resnet50_gn_timm":
                net = timm.create_model('resnet50_gn', pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == "vitsmall_timm":
                net = timm.create_model('vit_small_patch16_224', pretrained=True)
                args.lr = (0.0001 / 64) * bs
                # args.lr = (0.0002 / 64) * bs
            elif args.model == "vitbase_timm":
                net = timm.create_model('vit_base_patch16_224', pretrained=True)
                args.lr = (0.001 / 64) * bs 
            elif args.model == "convnext_tiny":
                net = timm.create_model('convnext_tiny_hnf', pretrained=True, checkpoint_path='/chenyaofo/cgh/cowork/cdy/research/checkpoints/hub/checkpoints/convnext_tiny_hnf_a2h-ab7e9df2.pth')
                args.lr = (0.000025 / 64) * bs
                # args.lr = (0.00025 / 64) * bs
            elif args.model == "swin_tiny":
                net = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
                args.lr = (0.00005 / 64) * bs
                # args.lr = (0.0005 / 64) * bs
            else:
                assert False, NotImplementedError
            net = net.cuda()
        else:
            assert False, NotImplementedError
        
        if args.exp_type in ["incorrect_labels_k1", "incorrect_labels_k5"]:
            top1_error_indices, top5_error_indices = create_error_grouped_datasets(
                error_val_loader, net, args, random_seed=2025, target_size=50000)
            if args.exp_type == "incorrect_labels_k1":
                print("使用no_adapt top-1错误样本进行实验")  
                selected_indices = top1_error_indices
            elif args.exp_type == "incorrect_labels_k5":
                print("使用no_adapt top-5错误样本进行实验")
                selected_indices = top5_error_indices
                  
            error_val_dataset.set_specific_subset(selected_indices)
            args.error_val_loader = error_val_loader
            print("After subset:")
            print(len(val_dataset), len(error_val_dataset))  
                
        double_lr_settings = {'bs1', 'label_shifts+bs1'}
        double_lr_methods = {'sar', 'sar_come', 'deyo', 'deyo_come', 'zerosiam'}
        if args.exp_type in double_lr_settings and args.method in double_lr_methods:
            args.lr *= 2
            logger.info(f"Double lr for method={args.method}, exp_type={args.exp_type}")
            
        print(f"lr is {args.lr}")
        

        if args.method == "tent":
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            tented_model = tent.Tent(net, optimizer)

            top1, top5, ece_avg = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adapttion accuracy of Tent is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())
            ece_list.append(ece_avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
            logger.info(f"ece_list are {ece_list}")
            

        elif args.method == "no_adapt":
            tented_model = net
            top1, top5, ece_avg  = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. Original Accuracy (no adapt) is top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
            
        elif args.method == 'deyo':
            net = deyo.configure_model(net)
            params, param_names = deyo.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
            deyo_model = deyo.DeYO(net, optimizer)
            
            top1, top5, ece_avg = validate(val_loader, deyo_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adapttion accuracy of DeYO is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())
            ece_list.append(ece_avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
            logger.info(f"ece_list are {ece_list}")

        elif args.method == "eata":
            # compute fisher informatrix
            args.corruption = 'original'
            fisher_dataset, fisher_loader = prepare_test_data(args)
            fisher_dataset.set_dataset_size(args.fisher_size)
            fisher_dataset.switch_mode(True, False)

            net = eata.configure_model(net)
            params, param_names = eata.collect_params(net)
            # fishers = None
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    targets = targets.cuda(args.gpu, non_blocking=True)
                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)

            top1, top5, ece_avg = validate(val_loader, adapt_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. After EATA Adapt: Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())
            ece_list.append(ece_avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
            logger.info(f"ece_list are {ece_list}")

        elif args.method == 'sar':
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)


            top1, top5, ece_avg = validate(val_loader, adapt_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adapttion accuracy of SAR is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())
            ece_list.append(ece_avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
            logger.info(f"ece_list are {ece_list}")

        elif args.method == 'deyo_come':
            net = deyo_come.configure_model(net)
            params, param_names = deyo_come.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
            deyo_model = deyo_come.DeYO(net, optimizer)
            
            top1, top5, ece_avg = validate(val_loader, deyo_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adapttion accuracy of DeYO-COME is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())
            ece_list.append(ece_avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
            logger.info(f"ece_list are {ece_list}")

        elif args.method == "zerosiam":
            net = BYOLWrapper(net, args.model)
            net = zerosiam.configure_model(net)
            params, param_names = zerosiam.collect_params(net)
            logger.info(param_names)
            backbone_optimizer = torch.optim.SGD(params, args.lr * args.lr_scale, momentum=0.9)
            predictor_optimizer = torch.optim.SGD([_ for _ in net.predictor.parameters()], args.lr * args.lr_p, momentum=0.9)
            # predictor_optimizer = torch.optim.SGD([_ for _ in net.predictor.parameters()], args.lr * args.lr_p * args.lr_scale, momentum=0.9)
            
            tented_model = zerosiam.Tent(net, backbone_optimizer, predictor_optimizer)
            tented_model.model_name = args.model
            top1, top5, ece_loss = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adapttion accuracy of ZeroSiam is top1 {top1:.5f} and top5: {top5:.5f} and ece_loss: {ece_loss*100:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())
            ece_list.append(ece_loss.item())

            logger.info(f"acc1s are {acc1s}, mean acc: {sum(acc1s)/len(acc1s)}")
            logger.info(f"acc5s are {acc5s}")
            logger.info(f"eces are {ece_list}")
            print(args.output)
        else:
            assert False, NotImplementedError
  
