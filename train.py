#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import time
import torch.nn.functional as F
import torch.nn as nn

from utils.dataset.isic import augmentation_rand, augmentation_randncls, augmentation_sim, augmentation_test
from utils.dataset.medical import Aptos_2019_train, Aptos_2019_test, Chromosome_abnormality_train, Chromosome_abnormality_test, Isic_2018_train, Isic_2018_test, Isic_2019_train, Isic_2019_test, HyperKvasir_train, HyperKvasir_test, HAM10000_train, HAM10000_test, APS_train, APS_test
from utils.eval_metrics import ConfusionMatrix, Auc
from models.IACL import IACL_model, CMNet
from models.loss import IACL, BCL
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import copy



'''function for saving model'''
def model_snapshot(model,new_modelpath,old_modelpath=None,only_bestmodel=False):
    if only_bestmodel and old_modelpath:
        os.remove(old_modelpath)
    torch.save(model.state_dict(),new_modelpath)


def get_feature_grad(grad):
    global features_grad
    features_grad = copy.copy(grad)


def main(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    log_file = open(os.path.join(args.log_path,'train_log.txt'), 'w')

    '''print args'''
    for arg in vars(args):
        print(arg, getattr(args, arg))
        print(arg, getattr(args, arg),file=log_file)

    if args.dataset == 'ISIC2018' or args.dataset == 'ISIC2019' or args.dataset == 'HAM10000':
        threshold = [1,4]
    elif args.dataset == 'APTOS2019':
        threshold = [1,2]
    elif args.dataset == 'HyperKvasir':
        threshold = [5,15]
    elif args.dataset == 'Chromosome_abnormality':
        threshold = [0,1]
    elif args.dataset == 'APS':
        threshold = [3,10]
     


    '''load models'''
    model = IACL_model(num_classes=args.num_classes,feat_dim=args.feat_dim)
    cov_newtwork = CMNet()



    if args.cuda:
        model.cuda()
        cov_newtwork.cuda()

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in cov_newtwork.parameters())/1000000.0),file=log_file)

    print("=============model init done=============")
    print("=============model init done=============",file=log_file)

 

    '''load optimizer'''
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_cov = optim.SGD(cov_newtwork.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)


    # cosine lr
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lr_scheduler_cov = optim.lr_scheduler.CosineAnnealingLR(optimizer_cov, T_max=args.epochs)

    complete = False

    '''load dataset''' 
    transfrom_train = [augmentation_rand, augmentation_sim]   
    #transfrom_train = [augmentation_randncls, augmentation_sim]   


    if args.dataset == 'ISIC2018':


        train_txt= args.data_path + "train.txt"
        val_txt= args.data_path + "val.txt"
        test_txt= args.data_path + "test.txt"

        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
        test_path = os.path.join(args.data_path, 'test')
        trainset = Isic_2018_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        valset = Isic_2018_test(path=train_path, txt=val_txt, transform=augmentation_test, class_map=trainset.class_map)
        testset = Isic_2018_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        valid_iterator = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == 'ISIC2019':
        train_txt= args.data_path + "train.txt"
        val_txt= args.data_path + "val.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
        test_path = os.path.join(args.data_path, 'test')
        trainset = Isic_2019_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        valset = Isic_2019_test(path=train_path, txt=val_txt, transform=augmentation_test, class_map=trainset.class_map)
        testset = Isic_2019_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        valid_iterator = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == 'HAM10000':
        train_txt= args.data_path + "train.txt"
        val_txt= args.data_path + "val.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
        trainset = HAM10000_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        valset = HAM10000_test(path=train_path, txt=val_txt, transform=augmentation_test, class_map=trainset.class_map)
        testset = HAM10000_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        valid_iterator = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == "APTOS2019":
        train_txt= args.data_path + "train.txt"
        val_txt= args.data_path + "val.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
        trainset = Aptos_2019_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        valset = Aptos_2019_test(path=train_path, txt=val_txt, transform=augmentation_test, class_map=trainset.class_map)
        testset = Aptos_2019_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        valid_iterator = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == "HyperKvasir":
        train_txt= args.data_path + "train.txt"
        val_txt= args.data_path + "val.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
       
        trainset = HyperKvasir_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        valset = HyperKvasir_test(path=train_path, txt=val_txt, transform=augmentation_test, class_map=trainset.class_map)
        testset = HyperKvasir_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        valid_iterator = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    else:
        raise ValueError("dataset error")
 


    '''load loss'''
    criterion_ce = nn.CrossEntropyLoss()
    
    criterion_IACL = IACL(feature_num=args.feat_dim, cls_num_list=args.cls_num_list)
    criterion_BCL = BCL(cls_num_list=args.cls_num_list)
    alpha = args.alpha
    beta = args.beta

    '''train'''
   
    best_acc = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

 
    old_model_path = None
    curr_patience = args.patience
    start_time = time.time()
    fc = model.fc
    per_cls_weights = None
    total_grad = 0
    out_cov = 0
    
    
   
    try:
        for e in range(args.epochs):
     
            model.train()
            cov_newtwork.train()
        
            print('Epoch:{}'.format(e),file=log_file)

            
            train_loss = 0.0

         
         
            ratio = ((e+1)/100)**2  
            start_time_epoch = time.time()
            for batch_index, (data, label) in enumerate(train_iterator):

                if args.cuda:
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                    label = label.cuda()

                diagnosis_label = label

                output, feat_, feat_mlp, feature = model(data) 
                feat_mlp = torch.cat([feat_mlp[0].unsqueeze(1), feat_mlp[1].unsqueeze(1)], dim=1)
                feat_non = torch.cat([feat_[0].unsqueeze(1), feat_[1].unsqueeze(1)], dim=1)
                original_features = torch.cat(torch.unbind(feat_non.clone(), dim=1), dim=0)
         
                original_features_1 = original_features.view(2*args.batch_size, original_features.shape[1], 1).expand(2*args.batch_size, original_features.shape[1], original_features.shape[1])
                original_features_2 = original_features_1.permute(0, 2, 1)
                matrix_feature = torch.bmm(original_features_1, original_features_2).reshape(2*args.batch_size, 1, original_features.shape[1], original_features.shape[1]).div(original_features.shape[1])

                out_cov = cov_newtwork(matrix_feature)       
                loss_ce = criterion_ce(output, diagnosis_label)
                loss_IACL = criterion_IACL(feat_non, label, e, out_cov, threshold)
                loss_BCL = criterion_BCL(feat_mlp, label)
               
                
                loss_CL = loss_BCL + ratio * loss_IACL
                loss = alpha * loss_ce + beta * loss_CL
  
                optimizer.zero_grad()
                optimizer_cov.zero_grad()

                loss.backward()
            
                optimizer.step()
                optimizer_cov.step()
         
                train_loss = train_loss + loss.item()
         

                if batch_index % 50 == 0 and batch_index != 0:
                    predicted_results = torch.argmax(output, dim=1)
                    correct_num = (predicted_results.cpu() == diagnosis_label.cpu()).sum().item()
                    acc = correct_num / len(diagnosis_label)
                    print('Training epoch: {} [{}/{}], Total Loss: {:.4f}, BCL Loss: {:.4f}, IACL Loss: {:.4f}, Accuracy: {:.4f}, Learning rate: {}'.format(e,
                        batch_index * args.batch_size, len(train_iterator.dataset), loss.item(), loss_BCL.item(), loss_IACL.item(), acc, optimizer.param_groups[0]['lr']))
                    print('Training epoch: {} [{}/{}], Total Loss: {:.4f}, BCL Loss: {:.4f}, IACL Loss: {:.4f}, Accuracy: {:.4f}, Learning rate: {}'.format(e,
                        batch_index * args.batch_size, len(train_iterator.dataset), loss.item(), loss_BCL.item(), loss_IACL.item(), acc, optimizer.param_groups[0]['lr']),file=log_file)
            lr_scheduler.step()
            lr_scheduler_cov.step()
            
            print("Epoch {} training time: {:.4f}s".format(e, time.time()-start_time_epoch))
            print("Epoch {} training time: {:.4f}s".format(e, time.time()-start_time_epoch),file=log_file)
            print("Epoch {} complete! Average Training loss: {:.4f}".format(e, train_loss / len(train_iterator)))
            print("Epoch {} complete! Average Training loss: {:.4f}".format(e, train_loss / len(train_iterator)),file=log_file)


            '''validation'''
            model.eval()
            cov_newtwork.eval()
            pro_diag, lab_diag = [], []
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('All_Acc@1', ':6.2f')
            head_top1 = AverageMeter('All_Acc@1', ':6.2f')
            major_top1 = AverageMeter('All_Acc@1', ':6.2f')
            tail_top1 = AverageMeter('All_Acc@1', ':6.2f')
            total_class_num=torch.zeros(args.num_classes).cuda()
            total_correct=torch.zeros(args.num_classes).cuda()


            running_loss = 0.0
            running_corrects = 0

            all_preds = torch.tensor([]).cuda()
            all_targets = torch.tensor([]).cuda()

            val_confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
            with torch.no_grad():
                for batch_index, (data, label) in enumerate(valid_iterator):
                    if args.cuda:
                        data = data.cuda()
                        label = label.cuda()
             
                    diagnosis_label = label

                    output, feat1 = model(data)
              
                    predicted_results = torch.argmax(output, dim=1)
                    pro_diag.extend(output.detach().cpu().numpy())
                    lab_diag.extend(diagnosis_label.cpu().numpy())

                    val_confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())

                    logit = output
                    _, preds = torch.max(logit, 1)
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, label.data))

                    running_loss += loss.item() * data.size(0)
                    running_corrects += torch.sum(preds == label.data)

                    acc1 = accuracy(output, label)
                    losses.update(loss.item(), data.size(0))
                    top1.update(acc1[0].item(), label.size(0))
                    _, predicted = output.max(1)
                   
                    target_one_hot = F.one_hot(label, num_classes=args.num_classes)
                    predict_one_hot = F.one_hot(predicted, num_classes=args.num_classes)
                    total_class_num = total_class_num + target_one_hot.sum(dim=0).to(torch.float)
                    total_correct = total_correct + (target_one_hot + predict_one_hot==2).sum(dim=0).to(torch.float)

                total_acc_classes = total_correct / total_class_num
                epoch_loss = running_loss / sum(total_class_num)
                epoch_acc = running_corrects / sum(total_class_num)

                
               
                dia_acc = val_confusion_diag.summary(log_file, e)
                Auc(pro_diag, lab_diag, args.num_classes, log_file)
                f_score_list = val_confusion_diag.get_f1score()


                if e > 0:
                    new_model_path = os.path.join(args.model_path, 'model_{}.pth'.format(e))
                    model_snapshot(model, new_model_path, old_modelpath=old_model_path, only_bestmodel=False)
                    old_model_path = new_model_path
                if dia_acc > best_acc:
                    curr_patience = args.patience
                    best_acc = dia_acc
                    new_model_path = os.path.join(args.model_path, 'bestacc_model_{}.pth'.format(e))
                    model_snapshot(model, new_model_path, old_modelpath=old_model_path, only_bestmodel=True)
                    old_model_path = new_model_path
                    print("Found new best model, saving to disk...")
                if e == args.epochs - 1:
                    print("Training complete, best accuracy: {:.4f}".format(best_acc))
                    print("Training complete, best accuracy: {:.4f}".format(best_acc),file=log_file)
                    complete = True
            log_file.flush()

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        log_file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def accuracy(output, target, topk=(1,)):
    """Computes Top-1 accuracy"""
    with torch.no_grad():
        maxk = 1
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = correct.float().sum() * (100.0 / batch_size)
        return [acc]

parser = argparse.ArgumentParser(description='Training for the classification task')

#dataset
parser.add_argument('--data_path', type=str, default='./data/ISIC2018/', help='the path of the data')
parser.add_argument('--dataset', type=str, default='ISIC2018',choices=['ISIC2018','ISIC2019','APTOS2019','Chromosome_abnormality','HAM10000','HyperKvasir','APS'], help='the name of the dataset')
parser.add_argument('--model_path', type=str, default="./Experiment/ISIC_CL/ISIC2018/test_git/", help='the path of the model')
parser.add_argument('--log_path', type=str, default=None, help='the path of the log')


# training parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--seed', type=int, default=1, help='random seed') 
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')

# loss weights
parser.add_argument('--alpha', type=float, default=2.0, choices=[0.5,1.0,2.0], help='weight of the cross entropy loss')
parser.add_argument('--beta', type=float, default=1.0, choices=[0.5,1.0,2.0], help='weight of the BHP loss')

# hyperparameters for model
parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=128)
parser.add_argument('--num_feature', type=int, default=2048)


def _seed_torch(args):
    r"""
    Sets custom seed for torch

    Args:
        - seed : Int

    Returns:
        - None

    """
    import random
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) 
        else:
            raise EnvironmentError("GPU device not found")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _seed_torch(args)
    if args.dataset == 'ISIC2018':
        args.cls_num_list = [4023, 667, 659, 308, 195, 84, 69]
        args.num_classes = 7
    elif args.dataset == 'ISIC2019':
        args.cls_num_list = [7725, 2712, 1993, 1574, 519, 376, 151, 143]
        args.num_classes = 8
    elif args.dataset == 'APTOS2019':
        args.cls_num_list = [1083, 599, 222, 177, 115]
        args.num_classes = 5
    elif args.dataset == 'HyperKvasir':
        args.cls_num_list = [688, 616, 605, 602, 599, 593, 560, 458, 388, 265, 241, 235, 156, 121, 81, 79, 31, 25, 21, 16, 7, 5, 4]
        args.num_classes = 23
    elif args.dataset == 'HAM10000':
        args.cls_num_list = [4642, 802, 783, 373, 137, 96, 89]
        args.num_classes = 7
    elif args.dataset == 'Chromosome_abnormality':
        args.cls_num_list = [347, 105, 67, 13]  
        args.num_classes = 4
    elif args.dataset == 'APS':
        args.cls_num_list = [1202, 780, 569, 564, 493, 446, 345, 291, 195, 157, 135, 61, 44, 40, 35, 22, 21]  
        args.num_classes = 17
    else:
        raise Exception("Invalid dataset name!")

    if args.log_path is None:
        args.log_path = args.model_path
    main(args)
    print("Done!")