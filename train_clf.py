import argparse
import torch
import numpy as np
import random
import torch.optim as optim
import time
from splitdata import split_train_test
from getgradients import get_gradients_set
from network import mlp
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CIFAR10_GRAD_ROOT = {'source':'./model/victim/vict-wrn28-10.pt',
                'distillation':'./model/attack/atta-d-wrn16-1.pt',
                'diff-architecture':'./model/attack/atta-wrn16-1.pt',
                'zero-shot':'./model/attack/atta-df-wrn16-1.pt',
                'fine-tune':'./model/attack/atta-finetune-wrn28-10.pt',
                'label-query':'./model/attack/atta-label-wrn16-1.pt',
                'logit-query':'./model/attack/atta-logit-wrn16-1.pt',
                'benign':'./model/benign/benign-wrn28-10.pt'}


IMAGENET_GRAD_ROOT = {'source':'./model/victim/vict-imgnet-resnet34.pt',
                'distillation':'./model/attack/atta-d-imgnet-resnet18.pt',
                'diff-architecture':'./model/victim/vict-imgnet-resnet18.pt',
                'zero-shot':'./model/attack/atta-df-imgnet-resnet18.pt',
                'fine-tune':'./model/attack/atta-finetune-imgnet-resnet34.pt',
                'label-query':'./model/attack/atta-label-imgnet-resnet18.pt',
                'logit-query':'./model/attack/atta-logit-imgnet-resnet18.pt',
                'benign':'./model/benign/benign-imgnet-resnet34.pt'}




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '--type', type=str, default='wrn28-10')
    parser.add_argument('--mlp_epoch', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', '--bs', type=int, default=8)
    parser.add_argument('--v_g_f_path', '--v', type=str, default='')
    parser.add_argument('--i_g_f_path', '--i', type=str, default='')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')

    args = parser.parse_args()
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.model_type == 'wrn28-10':
            args.v_g_f_path = './gradients_set/vict-wrn28-10.pt/cifar10_seurat_10%/g_f.npy'
            args.i_g_f_path = './gradients_set/benign-wrn28-10.pt/cifar10_seurat_10%/g_f.npy'
        elif args.model_type == 'wrn16-1':
            args.v_g_f_path = './gradients_set/vict-wrn16-1.pt/cifar10_seurat_10%/g_f.npy'
            args.i_g_f_path = './gradients_set/benign-wrn16-1.pt/cifar10_seurat_10%/g_f.npy'


    elif args.dataset == 'imagenet':
        args.num_classes = 20
        if args.model_type == 'resnet34-imgnet':
            args.v_g_f_path = './gradients_set/vict-imgnet-resnet34.pt/subimage_seurat_10%/g_f.npy'
            args.i_g_f_path = './gradients_set/benign-imgnet-resnet34.pt/subimage_seurat_10%/g_f.npy'
        elif args.model_type == 'resnet18-imgnet':
            args.v_g_f_path = './gradients_set/vict-imgnet-resnet18.pt/subimage_seurat_10%/g_f.npy'
            args.i_g_f_path = './gradients_set/benign-imgnet-resnet18.pt/subimage_seurat_10%/g_f.npy'
    else:
        raise('no such dataset')




    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    starttime = time.time()
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu != -1 else 'cpu'

    print('load gradients of victim and benign')
    # sign vector of gradients of victim model
    vict_g = np.sign(np.load(args.v_g_f_path))
    # sign vector of gradients of benign model
    benign_g = np.sign(np.load(args.i_g_f_path))  

    # split train & test set for training clf
    train_vict_g, test_vict_g, train_benign_g, test_benign_g = split_train_test(vict_g, benign_g)

    gradients_trainset, gradients_trainlabel = get_gradients_set(train_vict_g, train_benign_g)

    # gradients_testset, gradients_testlabel are used for testing clf on Victim model
    gradients_testset, gradients_testlabel = get_gradients_set(test_vict_g, test_benign_g)

    # train binary classifier with (benign_g, 0) and (vict_g, 1)
    print('train meta-classifier with sign vector of gradients')
    clf = mlp.MLP(len(gradients_trainset[0]), 2)
    clf = clf.to(device)
    print(clf)

    optimizer = optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    best_acc = 0

    for epoch in range(args.mlp_epoch):
        mlp.train(clf, gradients_trainset, gradients_trainlabel, epoch, optimizer, device)
        acc = mlp.test(clf, gradients_testset, gradients_testlabel, device, epoch)
        best_mlp_path = "model/clf/gradsign-%s-%s.pt" % (args.dataset, args.model_type)
        if acc > best_acc:
            best_acc = acc
            torch.save(clf.state_dict(), best_mlp_path)

    print("Test on Victim best acc=%.6f" % best_acc)

    print('Time cost: {} sec'.format(round(time.time() - starttime, 2)))