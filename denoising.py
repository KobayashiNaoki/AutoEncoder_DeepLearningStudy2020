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
    # transforms.Normalize((0.5,), (0.5,))
])
mnist_full = MNIST('./data', train=True, transform=transform, download=True)
mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
mnist_test = MNIST('./data', train=False, transform=transform, download=True)

train_iter = DataLoader(mnist_train, batch_size=200, shuffle=True)
valid_iter = DataLoader(mnist_val, batch_size=200)
test_iter = DataLoader(mnist_test, batch_size=200)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DenoisingStackAutoEncoder(nn.Module):
    def __init__(self, hidden, activation_fuctions=nn.Identity):
        super(DenoisingStackAutoEncoder, self).__init__()
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

    def forward(self, inputs, original):
        inputs = inputs.view(len(inputs), 28*28)
        original = original.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        loss = F.mse_loss(original, outputs)
        return loss

    def reconstruct(self, inputs, original):
        inputs = inputs.view(len(inputs), 28*28)
        original = original.view(len(inputs), 28*28)
        h = self.encoder(inputs)
        outputs = self.decoder(h)
        loss = F.mse_loss(original, outputs, reduction='none')
        outputs = outputs.view(len(inputs), 1, 28, 28)
        return outputs, h, loss


def training(epochs, model, noise_generator, optimizer, train_iter, valid_iter, device, save_all_epoch):
    for epoch in range(1, epochs+1):
        # train
        model.train()
        losses, n_samples = [], 0
        for batch in train_iter:
            inputs, labels = batch
            original = inputs
            inputs = original + noise_generator(original)
            batch_size = len(inputs)
            original = original.to(device)
            inputs = inputs.to(device)
            loss = model(inputs, original)
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
            original = inputs
            inputs = original + noise_generator(original)
            batch_size = len(inputs)
            original = original.to(device)
            inputs = inputs.to(device)
            with torch.no_grad():
                loss = model(inputs, original)

            losses.append(loss.item() * batch_size)
            n_samples += batch_size

        valid_loss = sum(losses) / n_samples

        print('epoch: {}\t train/loss: {:.4f}\t valid/loss: {:.4f}'.format(epoch, train_loss, valid_loss))

        if save_all_epoch:
            model_path = 'models/denoising/{}.hidden{}.epoch{}.pt'.format(model.__class__.__name__, HIDDEN_SIZE, N_EPOCHS)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }, model_path)
            print('saved at {}'.format(model_path))


model = DenoisingStackAutoEncoder(HIDDEN_SIZE, nn.Sigmoid)
noise_generator = AddGaussianNoise(0.5, 0.5)

print(model)
device = torch.device('cuda') if USE_GPU else torch.device('cpu')
model = model.to(device)

if MODEL_PATH is None:
    optimizer = torch.optim.Adam(model.parameters())
    training(N_EPOCHS, model, noise_generator, optimizer, train_iter, valid_iter, device, SAVE_ALL_EPOCH)
    if not SAVE_ALL_EPOCH:
        model_path = 'models/denoising/{}.hidden{}.epoch{}.pt'.format(model.__class__.__name__, HIDDEN_SIZE, N_EPOCHS)
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


def reconstruct(model, noise_generator, test_iter, device):
    samples = []
    model.eval()
    for batch in test_iter:
        inputs, labels = batch
        original = inputs
        inputs = original + noise_generator(original)
        batch_size = len(inputs)
        original = original.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs, hidden, loss = model.reconstruct(inputs, original)

        for i in range(batch_size):
            samples.append({
                'original': inputs[i][0].cpu().detach().numpy(),
                'reconstruct': outputs[i][0].cpu().detach().numpy(),
                'loss': loss[i],
                'embedding': hidden.cpu().detach().numpy(),
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


samples = reconstruct(model, noise_generator, test_iter, device)
model_name = model_path.split('/')[-1]
fig_path = 'figures/denoising/{}.png'.format(model_name.rstrip('.pt'))
plot(samples[:100], fig_path)
if not Path('figures/denoising/original.png').exists():
    plot(samples[:100], 'figures/denoising/original.png', original=True)
