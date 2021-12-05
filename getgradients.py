import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler


def collect_grads(grads_dict, avg_pool=False, pool_thresh=5000):
    g = []
    for grad in grads_dict:
        grad = np.asarray(grad.cpu())
        shape = grad.shape

        if len(shape) == 1:
           continue

        if len(shape) == 4:
            if shape[0] * shape[1] > pool_thresh:
                continue
            grad = grad.reshape(shape[0], shape[1], -1)

        if len(shape) > 2 or shape[0] * shape[1] > pool_thresh:
            if avg_pool:
                grad = np.mean(grad, -1)
            else:
                grad = np.max(grad, -1)

        g.append(grad.flatten())

    g = np.concatenate(g)
    return g


def get_gradients(model, train_loader, optimizer, device):
    # get gradient vector from model
    gradients_list = []
    model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        g = torch.autograd.grad(loss, model.parameters())
        g = collect_grads(g)
        gradients_list.append(g)
    return gradients_list


def get_gradients_set(g_p, g_np, norm=True, scale=True):
    g_trainset = np.vstack([g_p, g_np])
    g_label = np.concatenate([np.ones(len(g_p)), np.zeros(len(g_np))])

    if norm:
        normalizer = Normalizer(norm='l2')
        g_trainset = normalizer.transform(g_trainset)

    if scale:
        scaler = StandardScaler()
        g_trainset = scaler.fit_transform(g_trainset)

    return g_trainset, g_label


