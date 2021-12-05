import torch.nn as nn
import torch
import torch.utils.data as Data
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 8, input_size // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 16, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.softmax(out)
        return out


class MLP3(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP3, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size*2, input_size*4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 4, input_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size , output_size)

        )

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP4(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP4, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size*2, input_size*4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 4, input_size * 8),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 8, input_size*4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 4, input_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP5(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP5, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//4, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP2, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, input_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 8, input_size // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 16, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

def train(mlp, train_data, train_label, epoch, optimizer, device, batch_size=128):
    mlp.train()
    train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(train_label).long()
    torch_dataset = Data.TensorDataset(train_data, train_label)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    for step, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = mlp(batch_x)
        loss = F.cross_entropy(output, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch,  loss.item()))



def test(mlp, test_data, test_label, device, cur_epoch, batch_size=256):
    mlp.eval()
    test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label).long()
    torch_dataset = Data.TensorDataset(test_data, test_label)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loss = 0
    correct = 0
    subcorrect = list(0. for i in range(2))
    total = list(0. for i in range(2))
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = mlp(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            res = pred.eq(target.view_as(pred))
            # acc for subclass:
            for label_idx in range(len(target)):
                label_single = target[label_idx]
                subcorrect[label_single] += res[label_idx].item()
                total[label_single] += 1
        test_loss /= len(loader.dataset)
    if cur_epoch % 10 == 0:
        print('\nEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(cur_epoch, test_loss, correct, len(loader.dataset),100. * correct / len(loader.dataset)))
        for acc_idx in range(len(subcorrect)):
            try:
                acc = subcorrect[acc_idx] / total[acc_idx]
            except:
                acc = 0
            finally:
                print('laber {} acc: {:.2f}'.format(acc_idx, acc))
    return correct / len(loader.dataset)
