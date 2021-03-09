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
parser.add_argument('-stack', action='store_true')
parser.add_argument('-model', type=str, default=None)
parser.add_argument('--save-all', action='store_true')
args = parser.parse_args()

HIDDEN_SIZE = args.hidden
N_EPOCHS = args.epoch
USE_GPU = args.gpu
STACK = args.stack
MODEL_PATH = args.model
SAVE_ALL_EPOCH = args.save_all


transform = transforms.Compose([
    transforms.ToTensor(),
])
mnist_full = MNIST('./data', train=True, transform=transform, download=True)
mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
mnist_test = MNIST('./data', train=False, transform=transform, download=True)

train_iter = DataLoader(mnist_train, batch_size=200, shuffle=True)
valid_iter = DataLoader(mnist_val, batch_size=200)
test_iter = DataLoader(mnist_test, batch_size=200)


class AutoEncoder(nn.Module):
    def __init__(self, hidden, activation_fuctions=nn.Identity):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(28*28, hidden),
            activation_fuctions(),
        )

        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 28*28),
        )

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        loss = F.mse_loss(inputs, outputs)
        return loss

    def reconstruct(self, inputs):
        inputs = inputs.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        loss = F.mse_loss(inputs, outputs, reduction='none')
        outputs = outputs.view(len(inputs), 1, 28, 28)
        return outputs, h, loss


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
        loss = F.mse_loss(inputs, outputs)
        return loss

    def reconstruct(self, inputs):
        inputs = inputs.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        loss = F.mse_loss(inputs, outputs, reduction='none')
        outputs = outputs.view(len(inputs), 1, 28, 28)
        return outputs, h, loss


def training(epochs, model, optimizer, train_iter, valid_iter, device, save_all_epoch):
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

        # valid
        model.eval()
        losses, n_samples = [], 0
        for batch in valid_iter:
            inputs, labels = batch
            batch_size = len(inputs)
            inputs = inputs.to(device)
            with torch.no_grad():
                loss = model(inputs)

            losses.append(loss.item() * batch_size)
            n_samples += batch_size

        valid_loss = sum(losses) / n_samples

        print('epoch: {}\t train/loss: {:.4f}\t valid/loss: {:.4f}'.format(epoch, train_loss, valid_loss))

        if save_all_epoch:
            model_path = 'models/autoencoder/{}.hidden{}.epoch{}.pt'.format(model.__class__.__name__, HIDDEN_SIZE, N_EPOCHS)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }, model_path)
            print('saved at {}'.format(model_path))


if STACK:
    model = StackAutoEncoder(HIDDEN_SIZE, nn.Sigmoid)
else:
    model = AutoEncoder(HIDDEN_SIZE, nn.Sigmoid)

print(model)
device = torch.device('cuda') if USE_GPU else torch.device('cpu')
model = model.to(device)

if MODEL_PATH is None:
    optimizer = torch.optim.Adam(model.parameters())
    training(N_EPOCHS, model, optimizer, train_iter, valid_iter, device, SAVE_ALL_EPOCH)
    if not SAVE_ALL_EPOCH:
        model_path = 'models/autoencoder/{}.hidden{}.epoch{}.pt'.format(model.__class__.__name__, HIDDEN_SIZE, N_EPOCHS)
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
fig_path = 'figures/autoencoder/{}.png'.format(model_name.rstrip('.pt'))
plot(samples[:100], fig_path)
if not Path('figures/autoencoder/original.png').exists():
    plot(samples[:100], 'figures/autoencoder/original.png', original=True)


def plot_embedding(samples, save_path):
    plt.figure(figsize=(10, 8))
    embeddings = []
    labels = []
    for sample in samples:
        embeddings.append(sample['embedding'])
        labels.append(sample['gold_label'])

    embeddings = np.stack(embeddings)
    labels = np.stack(labels)
    sc = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='rainbow', alpha=0.6)
    plt.colorbar(sc)
    plt.savefig(save_path)
    print('save fig at {}'.format(save_path))
    return


if HIDDEN_SIZE == 2:
    fig_path = 'figures/autoencoder/{}.embedding.png'.format(model_name.rstrip('.pt'))
    plot_embedding(samples[:500], fig_path)
