# coding: utf-8
from __future__ import division
import argparse
import torch.optim as optim
import time
from modelloader import get_model
from getgradients import collect_grads
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import network
import os
import torch
import numpy as np
from scipy import stats
from scipy.stats import hmean
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CIFAR10_MODEL = {'source':'wrn28-10',
                'distillation':'wrn16-1',
                'zero-shot':'wrn16-1',
                'fine-tune':'wrn28-10',
                'label-query':'wrn16-1',
                'logit-query':'wrn16-1',
                'benign':'wrn28-10'}

IMAGENET_MODEL = {'source':'resnet34-imgnet',
                'distillation':'resnet18-imgnet',
                'zero-shot':'resnet18-imgnet',
                'fine-tune':'resnet34-imgnet',
                'label-query':'resnet18-imgnet',
                'logit-query':'resnet18-imgnet',
                'benign':'resnet34-imgnet'}



CIFAR10_MODEL_ROOT = {'source':'./model/victim/vict-wrn28-10.pt',
                'distillation':'./model/attack/atta-d-wrn16-1.pt',
                'zero-shot':'./model/attack/atta-df-wrn16-1.pt',
                'fine-tune':'./model/attack/atta-finetune-wrn28-10.pt',
                'label-query':'./model/attack/atta-label-wrn16-1.pt',
                'logit-query':'./model/attack/atta-logit-wrn16-1.pt',
                'benign':'./model/benign/benign-wrn28-10.pt'}

IMAGENET_MODEL_ROOT = {'source':'./model/victim/vict-imgnet-resnet34.pt',
                'distillation':'./model/attack/atta-d-imgnet-resnet18.pt',
                'zero-shot':'./model/attack/atta-df-imgnet-resnet18.pt',
                'fine-tune':'./model/attack/atta-finetune-imgnet-resnet34.pt',
                'label-query':'./model/attack/atta-label-imgnet-resnet18.pt',
                'logit-query':'./model/attack/atta-logit-imgnet-resnet18.pt',
                'benign':'./model/benign/benign-imgnet-resnet34.pt'}

NF_DATASET = {'cifar10':'./data/cifar10/',
              'imagenet':'./data/sub-image-20/'}

F_DATASET = {'cifar10':'./data/cifar10_seurat_10%/',
              'imagenet':'./data/subimage_seurat_10%/'}

CIFAR10_BENIGN_MODEL = {'source':'./model/benign/benign-wrn28-10.pt',
                'distillation':'./model/benign/benign-wrn16-1.pt',
                'zero-shot':'./model/benign/benign-wrn16-1.pt',
                'fine-tune':'./model/benign/benign-wrn28-10.pt',
                'label-query':'./model/benign/benign-wrn16-1.pt',
                'logit-query':'./model/benign/benign-wrn16-1.pt',
                'benign':'./model/benign/benign-wrn28-10.pt'}

IMAGENET_BENIGN_MODEL = {'source':'./model/benign/benign-imgnet-resnet34.pt',
                'distillation':'./model/benign/benign-imgnet-resnet18.pt',
                'zero-shot':'./model/benign/benign-imgnet-resnet18.pt',
                'fine-tune':'./model/benign/benign-imgnet-resnet34.pt',
                'label-query':'./model/benign/benign-imgnet-resnet18.pt',
                'logit-query':'./model/benign/benign-imgnet-resnet18.pt',
                'benign':'./model/benign/benign-imgnet-resnet34.pt'}

CIFAR10_CLF_ROOT = {'source':'./model/clf/gradsign-cifar10-wrn28-10.pt',
                'distillation':'./model/clf/gradsign-cifar10-wrn16-1.pt',
                'zero-shot':'./model/clf/gradsign-cifar10-wrn16-1.pt',
                'fine-tune':'./model/clf/gradsign-cifar10-wrn28-10.pt',
                'label-query':'./model/clf/gradsign-cifar10-wrn16-1.pt',
                'logit-query':'./model/clf/gradsign-cifar10-wrn16-1.pt',
                'benign':'./model/clf/gradsign-cifar10-wrn28-10.pt'}

IMAGENET_CLF_ROOT = {'source':'./model/clf/gradsign-imagenet-resnet34-imgnet.pt',
                'distillation':'./model/clf/gradsign-imagenet-resnet18-imgnet.pt',
                'zero-shot':'./model/clf/gradsign-imagenet-resnet18-imgnet.pt',
                'fine-tune':'./model/clf/gradsign-imagenet-resnet34-imgnet.pt',
                'label-query':'./model/clf/gradsign-imagenet-resnet18-imgnet.pt',
                'logit-query':'./model/clf/gradsign-imagenet-resnet18-imgnet.pt',
                'benign':'./model/clf/gradsign-imagenet-resnet34-imgnet.pt'}




def get_p_value(arrA, arrB):

    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a, b, equal_var=False)

    return p

def mult_test(prob_f, prob_nf, seed,  m, mult_num=40):
    p_list = []
    mu_list = []
    np.random.seed(seed)
    for t in range(mult_num):
        sample_num = m
        sample_list = [i for i in range(len(prob_f))]
        sample_list = random.sample(sample_list, sample_num)

        subprob_f = prob_f[sample_list]
        subprob_nf = prob_nf[sample_list]
        p_val = get_p_value(subprob_f, subprob_nf)
        p_list.append(p_val)
        mu_list.append(np.mean(subprob_f)-np.mean(subprob_nf))
    return p_list, mu_list


def load_img(dataset, path):
    if dataset == 'cifar10':
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == 'imagenet':
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    original = Image.open(path).convert('RGB')
    image = img_transform(original)
    image = np.asarray(image, dtype=np.float32)

    return image


def get_gradient_pair(sus_model, benign_model, device, args, optimizer_sus, optimizer_benign):
    sus_g_list = []
    benign_g_list = []
    sus_model.eval()
    benign_model.eval()

    for classid in range(args.num_classes):
        f_path = args.data_f_path +'/train/'+ str(classid) + '/'
        files = os.listdir(f_path)
        classid = torch.from_numpy(np.array([classid])).long()
        for file in files:
            filename, filetype = os.path.splitext(file)
            if filetype == '.jpeg' or filetype == '.jpg' or filetype == '.png' or filetype=='.JPEG':
                img_f = torch.from_numpy(load_img(args.dataset, f_path+filename+filetype))
                img_f = img_f.unsqueeze(0)
                img_f = img_f.to(device)
                classid = classid.to(device)

                # get gradients of img_f
                optimizer_sus.zero_grad()
                output_sus = sus_model(img_f)
                loss = F.cross_entropy(output_sus, classid)
                sus_g = torch.autograd.grad(loss, sus_model.parameters())
                sus_g = collect_grads(sus_g)
                sus_g_list.append(np.sign(sus_g))

                # get gradients of img_f
                optimizer_benign.zero_grad()
                output_benign = benign_model(img_f)
                loss = F.cross_entropy(output_benign, classid)
                benign_g = torch.autograd.grad(loss, benign_model.parameters())
                benign_g = collect_grads(benign_g)
                benign_g_list.append(np.sign(benign_g))

    return sus_g_list, benign_g_list

def get_prob_pair(clf, sus_g_list, benign_g_list, device):
    prob_f = []
    prob_nf = []
    for i in range(len(sus_g_list)):
        sus_g = torch.from_numpy(sus_g_list[i])
        benign_g = torch.from_numpy(benign_g_list[i])
        sus_g, benign_g = sus_g.to(device), benign_g.to(device)
        out_f = clf(sus_g)
        out_nf = clf(benign_g)
        prob_f.append(out_f.cpu().detach().numpy())
        prob_nf.append(out_nf.cpu().detach().numpy())
    return prob_f, prob_nf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='source', choices=['source','distillation','zero-shot','fine-tune','label-query','logit-query','benign'])
    parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'imagenet'])
    parser.add_argument('--model', '--suspicious_model_type',type=str, default='wrn28-10')
    parser.add_argument('--clf_model', type=str, default='mlp')
    parser.add_argument('--suspicious_model_root', '--s', type=str, default='./model/victim/vict-wrn28-10.pt',
                        help='suspicious model root')
    parser.add_argument('--benign_model_root', '--b', type=str, default='./model/benign/benign-wrn28-10.pt',
                        help='benign model root')
    parser.add_argument('--clf_root', '--c', type=str, default='./model/tracer/gradsign-cifar10-wrn28-10.pt',
                        help='clf model root')
    parser.add_argument('--batch_size', '--bs', type=int, default=1)
    parser.add_argument('--data_f_path', '--f', type=str, default='./data/cifar10_seurat_10%/')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.model = CIFAR10_MODEL[args.mode]
        args.suspicious_model_type = CIFAR10_MODEL[args.mode]
        args.suspicious_model_root = CIFAR10_MODEL_ROOT[args.mode]
        args.benign_model_root = CIFAR10_BENIGN_MODEL[args.mode]
        args.clf_root = CIFAR10_CLF_ROOT[args.mode]
        args.data_f_path = './data/cifar10_seurat_10%/'
    elif args.dataset == 'imagenet':
        args.num_classes = 20
        args.model = IMAGENET_MODEL[args.mode]
        args.suspicious_model_type = IMAGENET_MODEL[args.mode]
        args.suspicious_model_root = IMAGENET_MODEL_ROOT[args.mode]
        args.benign_model_root = IMAGENET_BENIGN_MODEL[args.mode]
        args.clf_root = IMAGENET_CLF_ROOT[args.mode]
        args.data_f_path = './data/subimage_seurat_10%/'
    else:
        raise('no such dataset')

    print(args)

    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu != -1 else 'cpu'

    random.seed(args.seed)
    starttime = time.time()

    # load suspicious model
    print('load suspicious model')
    sus_model = get_model(args)

    if device == 'cpu':
        sus_model.load_state_dict(torch.load(args.suspicious_model_root, map_location=torch.device('cpu')))
    else:
        sus_model.load_state_dict(torch.load(args.suspicious_model_root))
    sus_model.to(device)

    # load benign model
    print('load benign model')
    benign_model = get_model(args)

    if device == 'cpu':
        benign_model.load_state_dict(torch.load(args.benign_model_root, map_location=torch.device('cpu')))
    else:
        benign_model.load_state_dict(torch.load(args.benign_model_root))
    benign_model.to(device)

    # get output of suspicious (gradients or vd)
    print('get gradient from suspicious and benign model')
    optimizer_sus = optim.SGD(sus_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer_benign = optim.SGD(benign_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                               momentum=args.momentum)
    sus_g, benign_g = get_gradient_pair(sus_model, benign_model,device, args, optimizer_sus, optimizer_benign)


    # load meta-classifier
    if args.clf_model == 'mlp':
       clf = network.MLP(len(sus_g[0]), 2)
       if device == 'cpu':
           clf.load_state_dict(torch.load(args.clf_root, map_location=torch.device('cpu')))
       else:
           clf.load_state_dict(torch.load(args.clf_root))
       clf = clf.to(device)
       print(clf)
       clf.eval()

    # get probability from clf
    print('get probability from clf')
    prob_f, prob_nf = get_prob_pair(clf, sus_g, benign_g, device)
    model_name = args.suspicious_model_root.split('/')[3]
    model_name = model_name.split('.')[0]


    # T-test, get p-val
    prob_f = np.array(prob_f)
    prob_nf = np.array(prob_nf)

    #seed_start = 0
    seed = 100
    m = 10


    p_list, mu_list = mult_test(prob_f[:, 1], prob_nf[:, 1], seed=seed, m=m, mult_num=40)
    print('result:  p-val: {} mu: {}'.format(hmean(p_list), np.mean(mu_list)))

    print('Time cost: {} sec'.format(round(time.time() - starttime, 2)))
