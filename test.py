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

from utils.dataset.isic import augmentation_rand, augmentation_sim, augmentation_test
from utils.dataset.medical import Aptos_2019_train, Aptos_2019_test, Chromosome_abnormality_train, Chromosome_abnormality_test, Isic_2018_train, Isic_2018_test, Isic_2019_train, Isic_2019_test, HAM10000_train, HAM10000_test, HyperKvasir_train, HyperKvasir_test,APS_train,APS_test

from utils.eval_metrics import ConfusionMatrix, Auc
from sklearn.metrics import confusion_matrix
from models.IACL import IACL_model
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe


'''function for saving model'''
def model_snapshot(model,new_modelpath,old_modelpath=None,only_bestmodel=False):
    if only_bestmodel and old_modelpath:
        os.remove(old_modelpath)
    torch.save(model.state_dict(),new_modelpath)

'''function for getting proxies number'''

def main(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    log_file = open(os.path.join(args.log_path,'test.txt'), 'w')

    '''print args'''
    for arg in vars(args):
        print(arg, getattr(args, arg))
        print(arg, getattr(args, arg),file=log_file)


    '''load models'''
    model = IACL_model(num_classes=args.num_classes,feat_dim=args.feat_dim)

    if args.cuda:
        model.cuda()
      
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    print("=============model init done=============")
    print("=============model init done=============",file=log_file)


    '''load dataset'''
    transfrom_train = [augmentation_rand, augmentation_sim]
    if args.dataset == 'ISIC2018':

        print(args.data_path)
        train_txt= args.data_path + "train.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        test_path = os.path.join(args.data_path, 'test')
        trainset = Isic_2018_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        testset = Isic_2018_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)

        # test_iterator = DataLoader(isic2018_dataset(path=args.data_path, transform=augmentation_test, mode='test'),
        #                            batch_size=1, shuffle=False, num_workers=2)

    elif args.dataset == 'ISIC2019':
        train_txt= args.data_path + "train.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        test_path = os.path.join(args.data_path, 'test')
        trainset = Isic_2019_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        testset = Isic_2019_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        #train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
    elif args.dataset == "APTOS2019":
        train_txt= args.data_path + "train.txt"
        val_txt= args.data_path + "val.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')
        trainset = Aptos_2019_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        valset = Aptos_2019_test(path=train_path, txt=val_txt, transform=augmentation_test, class_map=trainset.class_map)
        testset = Aptos_2019_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        train_iterator = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
        valid_iterator = DataLoader(valset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
    elif args.dataset == 'HAM10000':
        train_txt= args.data_path + "train.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        test_path = os.path.join(args.data_path, 'test')
        trainset = HAM10000_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        testset = HAM10000_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        #train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
    elif args.dataset == 'HyperKvasir':
        train_txt= args.data_path + "train.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        test_path = os.path.join(args.data_path, 'test')
        trainset = HyperKvasir_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        testset = HyperKvasir_test(path=train_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        #train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
    elif args.dataset == "Chromosome_abnormality":
        print(args.data_path)
        train_txt= args.data_path + "train.txt"
        test_txt= args.data_path + "test.txt"
        train_path = os.path.join(args.data_path, 'train')
        test_path = os.path.join(args.data_path, 'test')
        trainset = Chromosome_abnormality_train(path=train_path, txt=train_txt, transform=transfrom_train, class_balance=False, num_classes=args.num_classes)
        testset = Chromosome_abnormality_test(path=test_path, txt=test_txt, transform=augmentation_test, class_map=trainset.class_map)

        #train_iterator = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
        test_iterator = DataLoader(testset, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
    else:
        raise ValueError("dataset error")

    try:
        # Testing
        
            best_epoch = 84
            #model_path = './results/ISIC-2019/model_'+str(i)+'.pth'
            model_path = './results/ISIC-2018/bestacc_model_'+str(best_epoch)+'.pth'
            #model_path = './results/HyperKvasir/model_'+str(i)+'.pth'
            #model_path = './results/HAM10000/model_'+str(i)+'.pth'
            #model_path = './results/APTOS2019/model_'+str(i)+'.pth'

            print(model_path)
            model.load_state_dict(torch.load(model_path),strict=True)
            model.eval()


            class_num = torch.zeros(args.num_classes).cuda()
            correct = torch.zeros(args.num_classes).cuda()
            top1 = AverageMeter('Acc@1', ':6.2f')
                

            all_feature = torch.empty((0, 2048)).cuda()
            all_target = torch.empty(0, dtype=torch.long).cuda()
            all_predict_target = torch.empty(0, dtype=torch.long).cuda()

            pro_diag, lab_diag = [], []
                
            confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
            with torch.no_grad():
                for batch_index, (data, label) in enumerate(test_iterator):
                    if args.cuda:
                        data = data.cuda()
                        label = label.cuda()
                    diagnosis_label = label
                    output, feat1 = model(data)

                    all_feature = torch.cat((all_feature, feat1), dim=0)
                    all_target = torch.cat((all_target, label), dim=0)

                    predicted_results = torch.argmax(output, dim=1)
                    pro_diag.extend(output.detach().cpu().numpy())
                    lab_diag.extend(diagnosis_label.cpu().numpy())
                    all_predict_target = torch.cat((all_predict_target, predicted_results), dim=0)
        
                    confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())
                    acc1 = accuracy(output, label, topk=(1,))
                    top1.update(acc1[0].item(), data.size(0))
                    _, predicted = output.max(1)
                    target_one_hot = F.one_hot(label, num_classes=args.num_classes)
                    predict_one_hot = F.one_hot(predicted, num_classes=args.num_classes)
                    class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
                    correct=correct + (target_one_hot + predict_one_hot==2).sum(dim=0).to(torch.float)

                print("Test confusion matrix:")
                print("Test confusion matrix:",file=log_file)
                confusion_diag.summary(log_file, best_epoch)
    
                acc_classes = correct / class_num
                cm = confusion_matrix(all_target.cpu().numpy(), all_predict_target.cpu().numpy())
                cm_np = np.array(cm)
        
                open("./results/"+args.dataset+"-confusion_matrix_test.txt","a+").write(" * epoch{i:d} \n Confusion Matrix:{CM:s} \n ".format(i=best_epoch,CM=str(cm_np)))

            if args.dataset == 'ISIC2018' or args.dataset == 'ISIC2019' or args.dataset == 'HAM10000':
                head_acc = acc_classes[0].mean()
                medium_acc = acc_classes[1:4].mean()
                tail_acc = acc_classes[4:].mean()
            elif args.dataset == 'APTOS2019':
                head_acc = acc_classes[0].mean()
                medium_acc = acc_classes[1].mean()
                tail_acc = acc_classes[2:].mean()
            elif args.dataset == 'HyperKvasir':
                head_acc = acc_classes[0:5].mean()
                medium_acc = acc_classes[5:15].mean()
                tail_acc = acc_classes[15:].mean()
            elif args.dataset == 'Chromosome_abnormality':
                head_acc = acc_classes[0:1].mean()
                medium_acc = acc_classes[1:2].mean()
                tail_acc = acc_classes[2:].mean()
                    
            print(" * epoch{i:d}  Acc@1 {top1.avg:.3f} HAcc {head_acc:.3f} MAcc {medium_acc:.3f} TAcc {tail_acc:.3f} \n".format(i=best_epoch,
                        top1=top1, head_acc=head_acc, medium_acc=medium_acc, tail_acc=tail_acc))
                        
            open("./results/"+args.dataset+"-test.txt","a+").write(" * epoch{i:d}  Acc@1 {top1.avg:.3f} HAcc {head_acc:.3f} MAcc {medium_acc:.3f} TAcc {tail_acc:.3f} \n".format(i=best_epoch,
                        top1=top1, head_acc=head_acc, medium_acc=medium_acc, tail_acc=tail_acc))
            # print(" * Acc@1 {top1.avg:.3f} HAcc {head_acc:.3f} MAcc {medium_acc:.3f} TAcc {tail_acc:.3f} \n".format(
            #     top1=top1, head_acc=head_acc, medium_acc=medium_acc, tail_acc=tail_acc),file=log_file)
        
        
            # all_feature = all_feature.cpu().detach().numpy()
            # all_target = all_target.cpu().detach().numpy()
            # #ts = TSNE(n_components=2, init='pca', random_state=0)
            # #result = ts.fit_transform(all_feature)
            # digits_final = TSNE(perplexity=30).fit_transform(all_feature)
            # name_train = 'test'
            # plot(digits_final, all_target, i)
            # name = 'test1' + '.png'
            # plt.savefig(fname=name)
            # TODO: this should also be done with the ProgressMeter

            # plt.figure(figsize=(16, 16))
            # plt.subplot(111)
            
            # all_feature = all_feature.cpu().detach().numpy()
            # all_target = all_target.cpu().detach().numpy()
            #     #ts = TSNE(n_components=2, init='pca', random_state=0)
            #     #result = ts.fit_transform(all_feature)
            # digits_final = TSNE(perplexity=30).fit_transform(all_feature)
            # name_train = 'test'
            # plot(digits_final, all_target, 1)
            # plt.grid(False)
            # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 移除x轴刻度线和标签
            # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # 移除y轴刻度线和标签
            # plt.box(False) 
            # name = 'ISIC2018-IACL-test1_'+str(i) + '.png'
            # plt.savefig(fname=name, dpi=1000)

    finally:
        log_file.close()


parser = argparse.ArgumentParser(description='Training for the classification task')

#dataset
parser.add_argument('--data_path', type=str, default='./data/ISIC2019/', help='the path of the data')
parser.add_argument('--dataset', type=str, default='ISIC2019',choices=['ISIC2018','ISIC2019','APTOS2019','HAM10000','HyperKvasir','Chromosome_abnormality'], help='the name of the dataset')
parser.add_argument('--model_path', type=str, default = './results/ISIC-2019/model_30.pth', help='the path of the model')
parser.add_argument('--log_path', type=str, default = None, help='the path of the log')


# training parameters
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')


# hyperparameters for model
parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=128)


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        maxk = 1
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        #for k in topk:
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        #print(res)
        return res


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
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        else:
            raise EnvironmentError("GPU device not found")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def plot(x, colors, epoch):
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    palette = np.array(sns.color_palette("hls", 7))

    # pastel, husl, and so on
    #print(palette.shape)
    # Create a scatter plot.
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    #co = (colors-9) / 10 
    co = colors
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
   
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[co.astype(np.int8)])
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[co.astype(np.int8)])
    sns.scatterplot(x=x[:,0], y=x[:,1], hue=colors, legend='full', palette=palette)
    # Add the labels for each digit.
    txts = []
    for i in range(8):  # class number
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
                #xtext, ytext = np.median(x[co == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.savefig('./digits_tsne-pastel'+str(epoch)+"t-SNE.png", dpi=1000)
    return f, ax, txts

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
        args.cls_num_list = [347, 105, 67, 13]  #
        args.num_classes = 4
    elif args.dataset == 'APS':
        args.cls_num_list = [1202, 780, 569, 564, 493, 446, 345, 291, 195, 157, 135, 61, 44, 40, 35, 22, 21]  #
        args.num_classes = 17
    else:
        raise Exception("Invalid dataset name!")

    if args.log_path is None:
        args.log_path = args.model_path
    main(args)
    print("Done!")