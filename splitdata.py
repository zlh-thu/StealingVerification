import numpy as np
from dataloader import get_cifar10
from dataloader import get_imagenet


def feature_data(args):
    # get feature embedded images 
    if args.dataset == 'cifar10':
        train_f_loader, _ = get_cifar10(args.data_f_path, args)
    elif args.dataset == 'imagenet':
        train_f_loader, _ = get_imagenet(args.data_f_path, args)
    return train_f_loader



def split_train_test(vict_g, benign_g, scale=0.8, shuffle=False):
    # split the train & test set for training clf
    indices_vict = np.arange(len(vict_g))
    indices_benign = np.arange(len(benign_g))

    indices_train_vict_g = np.random.choice(indices_vict, int(len(vict_g)*scale), replace=False)
    indices_test_vict_g = np.setdiff1d(indices_vict, indices_train_vict_g)

    indices_train_benign_g = np.random.choice(indices_benign, int(len(benign_g) * scale), replace=False)
    indices_test_benign_g = np.setdiff1d(indices_benign, indices_train_benign_g)

    if shuffle:
        np.random.shuffle(indices_train_vict_g)
        np.random.shuffle(indices_train_benign_g)

    train_vict_g = []
    test_vict_g = []
    train_benign_g = []
    test_benign_g = []

    test_num = min(len(indices_test_benign_g), len(indices_test_vict_g))

    for id in indices_train_vict_g:
        train_vict_g.append(vict_g[id])

    for id in indices_test_vict_g:
        test_vict_g.append(vict_g[id])

    for id in indices_train_benign_g:
        train_benign_g.append(benign_g[id])

    for id in indices_test_benign_g:
        test_benign_g.append(benign_g[id])

    test_vict_g = test_vict_g[:test_num]
    test_benign_g = test_benign_g[:test_num]

    return train_vict_g, test_vict_g, train_benign_g, test_benign_g
