from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10(path, args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.ImageFolder(root=path+'train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=1, drop_last=True) 

    test_dataset = datasets.ImageFolder(root=path+'test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    return train_loader, test_loader

def get_imagenet(path, args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(path+x, data_transforms[x])
                      for x in ['train', 'val', 'test']}
    train_loader = DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True,
                                   num_workers=4, drop_last=True)
    test_loader = DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=False,
                                 num_workers=4)
    return train_loader, test_loader

def get_dataloader(args):
    # get dataloader
    if args.dataset.lower()=='cifar10':
        return get_cifar10(args.data_path, args)
    elif args.dataset.lower()=='imagenet':
        return get_imagenet(args.data_path, args)