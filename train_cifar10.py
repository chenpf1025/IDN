import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from utils import get_softmax_out
from ops import train, train_soft, test
from networks.wideresnet import Wide_ResNet
from dataset import CIFAR10_soft
   
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 20)')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate (default: 0.1)')
    parser.add_argument('--dp', type=float, default=0.0, help='dropout rate (default: 0.0)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise_pattern', type=str, default='dependent', help='Noise pattern (default: dependent)')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate (default: 0.0)')
    parser.add_argument('--save', action='store_true', default=False, help='For saving softmax_out_avg')
    parser.add_argument('--SEAL', type=int, default=0, help='Phase of self-evolution')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = './data/CIFAR10'
    num_classes = 10
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
 
    train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    train_dataset_noisy = datasets.CIFAR10(root, train=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_test)

    targets_noisy = list(pd.read_csv(os.path.join('./data/CIFAR10/label_noisy', args.noise_pattern+str(args.noise_rate)+'.csv'))['label_noisy'].values.astype(int))
    train_dataset_noisy.targets = targets_noisy
    
    train_loader = torch.utils.data.DataLoader(train_dataset_noisy, batch_size=args.batch_size, shuffle=True, **kwargs)
    softmax_loader = torch.utils.data.DataLoader(train_dataset_noisy, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    
    def learning_rate(lr_init, epoch):
        optim_factor = 0
        if(epoch > 120):
            optim_factor = 2
        elif(epoch > 60):
            optim_factor = 1
        return lr_init*math.pow(0.2, optim_factor)

    # results
    results_root = os.path.join('results', 'cifar10_'+args.noise_pattern+str(args.noise_rate))
    if not os.path.isdir(results_root):
        os.makedirs(results_root)

    """ Get softmax_out_avg - normal training on noisy labels """
    if args.SEAL==0:
        # Building model
        model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=args.dp, num_classes=num_classes).to(device)

        # Training
        softmax_out_avg = np.zeros([len(train_dataset_noisy), num_classes])
        for epoch in range(1, args.epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
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
        train_dataset_soft = CIFAR10_soft(root, targets_soft=torch.Tensor(softmax_out_avg.copy()), train=True, transform=transform_train)
        train_dataset_soft.targets = targets_noisy
        train_loader_soft = torch.utils.data.DataLoader(train_dataset_soft, batch_size=args.batch_size, shuffle=True, **kwargs)

        # Building model
        model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=args.dp, num_classes=num_classes).to(device)

        # Training 
        softmax_out_avg = np.zeros([len(train_dataset_noisy), num_classes])
        for epoch in range(1, args.epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
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