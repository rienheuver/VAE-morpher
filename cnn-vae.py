from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision.utils import save_image
import datetime
import shutil
import signal
import random
import math
import vae_class
import matplotlib.pyplot as plt
import gaussian
from timeit import default_timer as timer

EXIT = False


def handler(signum, frame):
    global EXIT
    EXIT = True


signal.signal(signal.SIGINT, handler)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, help='pretrained model to use')
parser.add_argument('--debug', type=bool,
                    help='Enable debugging, graphs etc.', default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}

# Parameters
img_size = 160
max_channels = 256
latent_size = 256

train_data_path = 'normalised/training'
train_dataset = torchvision.datasets.ImageFolder(
    root=train_data_path,
    transform=torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

test_data_path = 'normalised/testing'
test_dataset = torchvision.datasets.ImageFolder(
    root=test_data_path,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

model = vae_class.VAE(max_channels, latent_size, img_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
start_epoch = 1

bce_losses = []
kld_losses = []
detail_losses = []

if args.model is not None:
    model, optimizer, start_epoch, bce_losses, kld_losses, detail_losses = vae_class.load_ckp(
        args.model, model, optimizer)

smoother = gaussian.GaussianSmoothing(3, 5, 25)


halfTensor = torch.ones(3,160,160).to('cuda') * 0.5
def loss_function(recon_x, x, mu, logvar, epoch, is_train=True):
    highpass_x = None
    for img in x:
        # f, axarr = plt.subplots(1,3)
        # axarr[0].axis('off')
        # axarr[0].set_title('Image')
        # axarr[1].axis('off')
        # axarr[1].set_title('Blurred')
        # axarr[2].axis('off')
        # axarr[2].set_title('Highpass')
        # axarr[0].imshow(img.permute(1,2,0).cpu().numpy())
        blur = img.view(1, 3, img_size, img_size)
        blur = F.pad(blur, (2, 2, 2, 2), mode='reflect')
        blur = smoother(blur)
        blur = blur.view(3, img_size, img_size)
        # axarr[1].imshow(blur.permute(1,2,0).cpu().numpy())
        highpassed = img - blur
        highpassed = highpassed
        # axarr[2].imshow(highpassed.permute(1,2,0).cpu().numpy())
        # plt.show()
        if highpass_x is None:
            highpass_x = highpassed
        elif (highpass_x.size()[0] == 1):
            highpass_x = highpass_x.stack(highpassed)
        else:
            highpass_x = torch.cat((highpass_x, highpassed), 0)
    
    highpass_recon_x = None
    for img in recon_x.view(-1, 3, img_size, img_size):
        # plt.imshow(img.permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        blur = img.view(1, 3, img_size, img_size)
        blur = F.pad(blur, (2, 2, 2, 2), mode='reflect')
        blur = smoother(blur)
        blur = blur.view(3, img_size, img_size)
        # plt.imshow(blur.permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        highpassed = img - blur
        highpassed = highpassed
        # plt.imshow(highpassed.permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        if highpass_recon_x is None:
            highpass_recon_x = highpassed
        elif (highpass_recon_x.size()[0] == 1):
            highpass_recon_x = highpass_recon_x.stack(highpassed)
        else:
            highpass_recon_x = torch.cat((highpass_recon_x, highpassed), 0)
    detailLoss = highpass_x.dist(highpass_recon_x) / args.batch_size * 3
    BCE = F.binary_cross_entropy(
        recon_x, x.view(-1, img_size**2), reduction='mean')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if is_train:
        bce_losses.append(BCE.item())
        kld_losses.append(KLD.item())
        detail_losses.append(detailLoss.item())

    KLDfactor = epoch / 1000
    KLDloss = 0.000001 * KLDfactor * KLD
    print('BCE ' + str(BCE.item()))
    print('KLD ' + str(KLD.item()) + ', calculated KLD: ' + str(KLDloss.item()))
    print('Detail ' + str(detailLoss.item()))

    return BCE + KLDloss + detailLoss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, something) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, epoch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'bce_losses': bce_losses,
        'kld_losses': kld_losses,
        'detail_losses': detail_losses
    }

    saveName = 'new-checkpoint.pt'
    if args.model:
        saveName = args.model
    vae_class.save_ckp(checkpoint, saveName)

    if args.debug:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        if epoch % 10 == 0:
            plt.plot(bce_losses[40:])
            plt.ylabel('BCE losses over time')
            plt.show()
            plt.plot(kld_losses[40:])
            plt.ylabel('KL losses over time')
            plt.show()

            print('avg bce', sum(bce_losses) / len(bce_losses))
            print('avg kld', sum(kld_losses) / len(kld_losses))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model.forward(data)
            test_loss += loss_function(recon_batch,
                                       data, mu, logvar, 0, is_train=False).item()
            if i == 0 and epoch % 10 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 3, img_size, img_size)[:n]])
                save_image(comparison.cpu(),
                           'reconstructions/epoch' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    if args.debug:
        print('====> Epoch: {} Testset loss: {:.4f}'.format(epoch, test_loss))


if __name__ == "__main__":
    for epoch in range(start_epoch, args.epochs + 1):
        startTime = timer()
        train(epoch)
        test(epoch)
        endTime = timer()
        print('Epoch duration: ' + str(endTime - startTime))
        print('Epoch: {}, KL loss: {}, BCE loss: {}, Detail loss: {}'.format(
            epoch, kld_losses[-1], bce_losses[-1], detail_losses[-1]))
        if EXIT:
            break
        with torch.no_grad():
            if epoch % 1 == 0:
                multiLineup = None
                for i in range(0, 10, 1):
                    index1 = (i+135) % len(test_loader.dataset)
                    index2 = (i+135+1) % len(test_loader.dataset)
                    face1, _ = test_loader.dataset[index1]
                    face2, _ = test_loader.dataset[index2]
                    mu1, logvar1 = model.encode(
                        face1.unsqueeze(0).to(device))
                    L1 = model.reparameterize(mu1, logvar1, 0)
                    mu2, logvar2 = model.encode(
                        face2.unsqueeze(0).to(device))
                    L2 = model.reparameterize(mu2, logvar2, 0)
                    Lnew = (L1 + L2) / 2

                    reconstruct1 = model.decode(L1).cpu()
                    reconstruct2 = model.decode(L2).cpu()
                    morph = model.decode(Lnew).cpu()

                    lineup = torch.cat(
                        (face1.view(3, img_size, img_size),
                         reconstruct1.view(3, img_size, img_size),
                         morph.view(3, img_size, img_size),
                         reconstruct2.view(3, img_size, img_size),
                         face2.view(3, img_size, img_size)), 2)
                    if multiLineup is None:
                        multiLineup = lineup
                    else:
                        multiLineup = torch.cat((multiLineup, lineup), 1)
                    # save_image(morph, 'single-morphs/morph' + str(i) + '.png')
                save_image(multiLineup,
                           'morphs/epoch' + str(epoch) + '.png')
