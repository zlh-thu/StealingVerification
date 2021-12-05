from network import *

def get_model(args):
    # init model
    if args.model.lower() == 'vgg19':
        return VGG('VGG19')
    elif args.model.lower() == 'wrn28-10':
        return Wide_ResNet(28, 10, 0.3, args.num_classes)
    elif args.model.lower() == 'wrn16-1':
        return Wide_ResNet(16, 1, 0.3, args.num_classes)
    elif args.model.lower() == 'resnet18-imgnet':
        return imagenet_get_model('res18')
    elif args.model.lower() == 'resnet34-imgnet':
        return imagenet_get_model('res34')

def load_model(args):
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_root))
