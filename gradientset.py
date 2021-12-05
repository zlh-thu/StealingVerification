import argparse
import torch
import numpy as np
import random
import torch.optim as optim
import time
from splitdata import feature_data
from modelloader import get_model
from getgradients import get_gradients
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--model_root', '--m', type=str, default='./model/victim/backdoor-resnet34-cifar10.pth',
                        help='model root')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', '--bs',type=int, default=1)
    parser.add_argument('--data_f_path', '--f', type=str, default='./data/cifar10_seurat_small/')
    parser.add_argument('--gradientset_path', type=str, default='./gradients_set/')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.data_f_path = './data/cifar10_seurat_10%/'


    elif args.dataset == 'imagenet':
        args.num_classes = 20
        args.data_f_path = './data/subimage_seurat_10%/'
    else:
        raise('no such dataset')

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.data_f_path = './data/cifar10_seurat_10%/'
    elif args.dataset == 'imagenet':
        args.num_classes = 20
        args.data_f_path = './data/subimage_seurat_10%/'
    else:
        raise('no such dataset')




    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.batch_size = 1

    start_time = time.time()
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu != -1 else 'cpu'

    print('load model')
    model = get_model(args)
    model.to(device)

    if device == 'cpu':
        model.load_state_dict(torch.load(args.model_root, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.model_root))
    


    print('get gradients from model')
    # get feature embedded images
    train_f_loader = feature_data(args)

    # get gradients from model
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    g_f = get_gradients(model, train_f_loader, optimizer, device)
    g_f = np.asarray(g_f)

    print('save gradients')
    model_name = args.model_root.split('/')[3]
    f_data_name = args.data_f_path.split('/')[2]
    g_f_path = args.gradientset_path + model_name + '/' + f_data_name

    isExists = os.path.exists(g_f_path)
    if not isExists:
        os.makedirs(g_f_path)

    np.save(g_f_path+'/g_f.npy', g_f)

    print('Time cost: {} sec'.format(round(time.time() - start_time, 2)))