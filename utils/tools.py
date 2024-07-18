from torch.nn.modules import loss
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch.nn as nn
from itertools import repeat


class Spatial_Dropout(nn.Module):
    def __init__(self, drop_prob):

        super(Spatial_Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self, input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))


def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + 1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def R2(pred, true):
    return r2_score(true, pred)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    return mae,mse,rmse,mape,mspe,r2

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class TopkMSELoss(torch.nn.Module):
    def __init__(self, topk) -> None:
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, output, label):
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]

        return losses

class SingleStepLoss(torch.nn.Module):
    """ Compute top-k log-likelihood and mse. """

    def __init__(self, ignore_zero):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, mu, sigma, labels, topk=0):
        if self.ignore_zero:
            indexes = (labels != 0)
        else:
            indexes = (labels >= 0)

        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])

        diff = labels[indexes] - mu[indexes]
        se = diff * diff

        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]

        return likelihood, se

def AE_loss(mu, labels, ignore_zero):
    if ignore_zero:
        indexes = (labels != 0)
    else:
        indexes = (labels >= 0)

    ae = torch.abs(labels[indexes] - mu[indexes])
    return ae

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        # lr_adjust = {epoch: args.learning_rate}
           lr_adjust = {epoch: args.learning_rate * (0.75 ** ((epoch - 1) // 1))}
        # lr_adjust = {epoch: args.learning_rate * (0.2 ** (epoch // 2))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def visual(true, preds=None, name='./img/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
