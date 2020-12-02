import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from utils import get_softmax_out
from ops import train, train_soft, test
from dataset import MNIST_soft
from networks.cnn_mnist import MNIST_CNN
 
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 200)')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise_pattern', type=str, default='dependent', help='Noise pattern (default: dependent)')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate (default: 0.0)')
    parser.add_argument('--save', action='store_true', default=False, help='For saving softmax_out_avg')
    parser.add_argument('--SEAL', type=int, default=0, help='Phase of self-evolution')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = './data'
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #0.1307, 0.3081 are the mean and std of mnist
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    train_dataset_noisy = datasets.MNIST(root, train=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, transform=transform)

    targets_noisy = torch.Tensor(pd.read_csv(os.path.join('./data/MNIST/label_noisy', args.noise_pattern+str(args.noise_rate)+'.csv'))['label_noisy'].values.astype(int))
    train_dataset_noisy.targets = targets_noisy
    
    train_loader = torch.utils.data.DataLoader(train_dataset_noisy, batch_size=args.batch_size, shuffle=True, **kwargs)
    softmax_loader = torch.utils.data.DataLoader(train_dataset_noisy, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # results
    results_root = os.path.join('results', 'mnist_'+args.noise_pattern+str(args.noise_rate))
    if not os.path.isdir(results_root):
        os.makedirs(results_root)

    """ Get softmax_out_avg - normal training on noisy labels """
    if args.SEAL==0:
        # Building model
        model = MNIST_CNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Training
        softmax_out_avg = np.zeros([len(train_dataset_noisy), 10])
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)

        softmax_out_avg /= args.epochs
        if args.save:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_avg_'+args.noise_pattern+str(args.noise_rate)+'_normal.npy')
            np.save(softmax_root, softmax_out_avg)
            print('new softmax_out_avg saved to', softmax_root, ', shape: ', softmax_out_avg.shape)

    """ Self Evolution - training on softmax_out_avg """
    if args.SEAL>=1:
        # Loading softmax_out_avg of last phase
        if args.SEAL==1:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_avg_'+args.noise_pattern+str(args.noise_rate)+'_normal.npy')
        else:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_avg_'+args.noise_pattern+str(args.noise_rate)+'_SEAL'+str(args.SEAL-1)+'.npy')
        softmax_out_avg = np.load(softmax_root)
        print('softmax_out_avg loaded from', softmax_root, ', shape: ', softmax_out_avg.shape)
       
        # Dataset with soft targets
        train_dataset_soft = MNIST_soft(root, targets_soft=torch.Tensor(softmax_out_avg.copy()), train=True, transform=transform)
        train_dataset_soft.targets = targets_noisy
        train_loader_soft = torch.utils.data.DataLoader(train_dataset_soft, batch_size=args.batch_size, shuffle=True, **kwargs)

        # Building model
        model = MNIST_CNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Training 
        softmax_out_avg = np.zeros([len(train_dataset_noisy), 10])
        for epoch in range(1, args.epochs + 1):
            train_soft(args, model, device, train_loader_soft, optimizer, epoch)
            test(args, model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)

        softmax_out_avg /= args.epochs
        if args.save:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_avg_'+args.noise_pattern+str(args.noise_rate)+'_SEAL'+str(args.SEAL)+'.npy')
            np.save(softmax_root, softmax_out_avg)
            print('new softmax_out_avg saved to', softmax_root, ', shape: ', softmax_out_avg.shape)
     
if __name__ == '__main__':
    main()