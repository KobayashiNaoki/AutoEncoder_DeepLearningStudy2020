import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from pathlib import Path
import argparse

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('-hidden', type=int, default=20)
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-gpu', action='store_true')
parser.add_argument('-model', type=str, default=None)
parser.add_argument('--save-all', action='store_true')
args = parser.parse_args()

HIDDEN_SIZE = args.hidden
N_EPOCHS = args.epoch
USE_GPU = args.gpu
MODEL_PATH = args.model
SAVE_ALL_EPOCH = args.save_all


transform = transforms.Compose([
    transforms.ToTensor(),
])
valid_label = [2, 7]
mnist_full = MNIST('./data', train=True, transform=transform, download=True)
mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
mnist_test = MNIST('./data', train=False, transform=transform, download=True)

mnist_train_subset = []
for sample in mnist_train:
    image, label = sample
    if label in valid_label:
        mnist_train_subset.append(sample)

mnist_valid_subset_true = []
for sample in mnist_train:
    image, label = sample
    if label in valid_label:
        mnist_valid_subset_true.append(sample)
mnist_valid_subset_false = []
for sample in mnist_train:
    image, label = sample
    if label not in valid_label:
        mnist_valid_subset_false.append(sample)

train_iter = DataLoader(mnist_train_subset, batch_size=200, shuffle=True)
valid_iter_true = DataLoader(mnist_valid_subset_true, batch_size=200)
valid_iter_false = DataLoader(mnist_valid_subset_false, batch_size=200)
valid_iters = [valid_iter_true, valid_iter_false]
valid_iter = DataLoader(mnist_val, batch_size=200)
test_iter = DataLoader(mnist_test, batch_size=200)


class StackAutoEncoder(nn.Module):
    def __init__(self, hidden, activation_fuctions=nn.Identity):
        super(StackAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(28*28, 500),
            activation_fuctions(),
            nn.Linear(500, 250),
            activation_fuctions(),
            nn.Linear(250, hidden),
            activation_fuctions(),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 250),
            activation_fuctions(),
            nn.Linear(250, 500),
            activation_fuctions(),
            nn.Linear(500, 28*28),
        )

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        # loss = F.mse_loss(inputs, outputs)
        loss = F.mse_loss(inputs, outputs, reduction='none').mean(dim=1).mean(dim=0)
        return loss

    def reconstruct(self, inputs):
        inputs = inputs.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        loss = F.mse_loss(inputs, outputs, reduction='none').mean(dim=1)
        outputs = outputs.view(len(inputs), 1, 28, 28)
        return outputs, h, loss


def training(epochs, model, optimizer, train_iter, valid_iters, device, save_all_epoch):
    for epoch in range(1, epochs+1):
        # train
        model.train()
        losses, n_samples = [], 0
        for batch in train_iter:
            inputs, labels = batch
            batch_size = len(inputs)
            inputs = inputs.to(device)
            loss = model(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item() * batch_size)
            n_samples += batch_size

        train_loss = sum(losses) / n_samples

        # valid (true case)
        model.eval()
        losses, n_samples = [], 0
        for batch in valid_iters[0]:
            inputs, labels = batch
            batch_size = len(inputs)
            inputs = inputs.to(device)
            with torch.no_grad():
                loss = model(inputs)

            losses.append(loss.item() * batch_size)
            n_samples += batch_size

        valid_loss_true = sum(losses) / n_samples

        # valid (false case)
        losses, n_samples = [], 0
        for batch in valid_iters[1]:
            inputs, labels = batch
            batch_size = len(inputs)
            inputs = inputs.to(device)
            with torch.no_grad():
                loss = model(inputs)

            losses.append(loss.item() * batch_size)
            n_samples += batch_size

        valid_loss_false = sum(losses) / n_samples
        print('epoch: {}\t train/loss: {:.4f}\t valid/true/loss: {:.4f}\t valid/false/loss: {:.4f}'.format(
            epoch, train_loss, valid_loss_true, valid_loss_false))

        if save_all_epoch:
            model_path = 'models/outlier_detection/{}.hidden{}.epoch{}.pt'.format(model.__class__.__name__, HIDDEN_SIZE, N_EPOCHS)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }, model_path)
            print('saved at {}'.format(model_path))


model = StackAutoEncoder(HIDDEN_SIZE, nn.Sigmoid)

print(model)
device = torch.device('cuda') if USE_GPU else torch.device('cpu')
model = model.to(device)

if MODEL_PATH is None:
    optimizer = torch.optim.Adam(model.parameters())
    training(N_EPOCHS, model, optimizer, train_iter, valid_iters, device, SAVE_ALL_EPOCH)
    if not SAVE_ALL_EPOCH:
        model_path = 'models/outlier_detection/{}.hidden{}.epoch{}.pt'.format(model.__class__.__name__, HIDDEN_SIZE, N_EPOCHS)
        torch.save({
            'epoch': N_EPOCHS,
            'model': model.state_dict(),
            'optim': optimizer.state_dict()
        }, model_path)
        print('saved at {}'.format(model_path))
else:
    model_path = MODEL_PATH
    model_state = torch.load(model_path)
    model.load_state_dict(model_state['model'])
    print('load from {}'.format(model_path))


def reconstruct(model, test_iter, device):
    samples = []
    model.eval()
    for batch in test_iter:
        inputs, labels = batch
        batch_size = len(inputs)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs, hidden, loss = model.reconstruct(inputs)

        for i in range(batch_size):
            samples.append({
                'original': inputs[i][0].cpu().detach().numpy(),
                'reconstruct': outputs[i][0].cpu().detach().numpy(),
                'loss': loss[i],
                'embedding': hidden[i].cpu().detach().numpy(),
                'gold_label': labels[i].item(),
            })

    return samples


def plot(samples, save_path, original=False):
    plt.figure(figsize=(10, 10))
    for i, sample in enumerate(samples, 1):
        plt.subplot(10, 10, i)
        image = sample['original'] if original else sample['reconstruct']
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.savefig(save_path)
    print('save fig at {}'.format(save_path))
    return


samples = reconstruct(model, test_iter, device)
model_name = model_path.split('/')[-1]
fig_path = 'figures/outlier_detection/{}.png'.format(model_name.rstrip('.pt'))
plot(samples[:100], fig_path)
if not Path('figures/outlier_detection/original.png').exists():
    plot(samples[:100], 'figures/outlier_detection/original.png', original=True)


def outlier_detection(samples, threshold):
    TP, TN, FP, FN = 0, 0, 0, 0
    for sample in samples:
        if sample['loss'] < threshold:
            # the trained model predicts this sample is not outlier
            if sample['gold_label'] in valid_label:
                # model: not outlier, gold: not outlier
                TP += 1
            else:
                # model: not outlier, gold: outlier
                FN += 1
        else:
            if sample['gold_label'] in valid_label:
                # model: outlier, gold: not outlier
                FP += 1
            else:
                # model: outlier, gold: outlier
                TN += 1

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    recall = TP / (TP + FP) if (TP+FP) != 0 else 0
    precision = TP / (TP + FN) if (TP+FN) != 0 else 0
    if recall + precision != 0:
        f_measure = 2*recall*precision / (recall + precision)
    else:
        f_measure = 0

    print('threshold: {:.4f}\t acc: {:.4f}\t P: {:.4f}\t R: {:.4f}\t F: {:.4f}'.format(
        threshold, accuracy, precision, recall, f_measure))
    return f_measure


# tuning
samples_val = reconstruct(model, valid_iter, device)
best_threshold, best_f = 0, -1
print('val results (tuning of threshold)')
for threshold in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07]:
    f = outlier_detection(samples_val, threshold)
    if best_f < f:
        best_threshold = threshold
        best_f = f

print('test results')
outlier_detection(samples, best_threshold)
