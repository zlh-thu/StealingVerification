# Defending against Model Stealing Attacks via Verifying Embedded External Features

This is the official implementation of our paper [Defending against Model Stealing Attacks via
Verifying Embedded External Features](), accepted by the AAAI Conference on Artificial Intelligence (AAAI), 2022. This research project is developed based on Python 3 and Pytorch, created by [Yiming Li](http://liyiming.tech/) and Linghui Zhu.



## Pipeline
![Pipeline](https://github.com/zlh-thu/StealingVerification/blob/main/images/pipeline.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Make sure the directory follows:
```File Tree
stealingverification
├── data
│   ├── cifar10
│   └── ...
├── gradients_set 
│   
├── prob
│   
├── network
│   
├── model
│   ├── victim
│   └── ...
|
```


## Dataset Preparation
Make sure the directory ``data`` follows:
```File Tree
data
├── cifar10_seurat_10%
|   ├── train
│   └── test
├── cifar10  
│   ├── train
│   └── test
├── subimage_seurat_10%
│   ├── train
|   ├── val
│   └── test
├── sub-imagenet-20
│   ├── train
|   ├── val
│   └── test
```


>📋  Data Download Link:  
>[data](https://www.dropbox.com/sh/wzvq37j3fqaxzdj/AACZO-U4P5LVCANaEE8v7DIna?dl=0)


## Model Preparation
Make sure the directory ``model`` follows:
```File Tree
model
├── victim
│   ├── vict-wrn28-10.pt
│   └── ...
├── benign
│   ├── benign-wrn28-10.pt
│   └── ...
├── attack
│   ├── atta-label-wrn16-1.pt
│   └── ...
└── clf
```

>📋  Model Download Link:  
>[model](https://www.dropbox.com/sh/w3hgjlranvjifk8/AABCjMKxmqHQzSro0rZuLn3Ia?dl=0)



## Collecting Gradient Vectors
Collect gradient vectors of victim and benign model with respect to transformed images.

CIFAR-10:
```Collect
python gradientset.py --model=wrn16-1 --m=./model/victim/vict-wrn16-1.pt --dataset=cifar10 --gpu=0
python gradientset.py --model=wrn28-10 --m=./model/victim/vict-wrn28-10.pt --dataset=cifar10 --gpu=0
python gradientset.py --model=wrn16-1 --m=./model/benign/benign-wrn16-1.pt --dataset=cifar10 --gpu=0
python gradientset.py --model=wrn28-10 --m=./model/benign/benign-wrn28-10.pt --dataset=cifar10 --gpu=0
```

ImageNet:
```Collect
python gradientset.py --model=resnet34-imgnet --m=./model/victim/vict-imgnet-resnet34.pt --dataset=imagenet --gpu=0
python gradientset.py --model=resnet18-imgnet --m=./model/victim/vict-imgnet-resnet18.pt --dataset=imagenet --gpu=0
python gradientset.py --model=resnet34-imgnet --m=./model/benign/benign-imgnet-resnet34.pt --dataset=imagenet --gpu=0
python gradientset.py --model=resnet18-imgnet --m=./model/benign/benign-imgnet-resnet18.pt --dataset=imagenet --gpu=0
```

## Training Ownership Meta-Classifier

To train the ownership meta-classifier in the paper, run these commands:

CIFAR-10:
```train
python train_clf.py --type=wrn28-10 --dataset=cifar10 --gpu=0
python train_clf.py --type=wrn16-1 --dataset=cifar10 --gpu=0
```
ImageNet:
```train
python train_clf.py --type=resnet34-imgnet --dataset=imagenet --gpu=0
python train_clf.py --type=resnet18-imgnet --dataset=imagenet --gpu=0
```

## Ownership Verification

To verify the ownership of the suspicious models, run this command:

CIFAR-10:
```Verification
python ownership_verification.py --mode=source --dataset=cifar10 --gpu=0 

#mode: ['source','distillation','zero-shot','fine-tune','label-query','logit-query','benign']
```

ImageNet:
```Verification
python ownership_verification.py --mode=logit-query --dataset=imagenet --gpu=0 

#mode: ['source','distillation','zero-shot','fine-tune','label-query','logit-query','benign']
```
## An Example of the Result
```Verification
python ownership_verification.py --mode=fine-tune --dataset=cifar10 --gpu=0 

result:  p-val: 1.9594572166549425e-08 mu: 0.47074130177497864
```

## Reference
If our work or this repo is useful for your research, please cite our paper as follows:
```
@inproceedings{li2022defending,
  title={Defending against Model Stealing via Verifying Embedded External Features},
  author={Li, Yiming and Zhu, Linghui and Jia, Xiaojun and Jiang, Yong and Xia, Shu-Tao and Cao, Xiaochun},
  booktitle={AAAI},
  year={2022}
}
```

