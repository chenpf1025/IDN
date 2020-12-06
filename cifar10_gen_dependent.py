import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from utils import get_softmax_out
from ops import train, test
from networks.wideresnet import Wide_ResNet
 
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 150)')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate (default: 0.1)')
    parser.add_argument('--dp', type=float, default=0.0, help='dropout rate (default: 0.3)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate (default: 0.0)')
    parser.add_argument('--load', action='store_true', default=False, help='Load existing averaged softmax')
    parser.add_argument('--gen', action='store_true', default=False, help='Generate noisy labels')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = './data/CIFAR10'
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
 
    train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    def learning_rate(lr_init, epoch):
        optim_factor = 0
        if(epoch > 120):
            optim_factor = 2
        elif(epoch > 60):
            optim_factor = 1
        return lr_init*math.pow(0.2, optim_factor)

    if args.load:
        softmax_out_avg = np.load('data/CIFAR10/label_noisy/softmax_out_avg.npy')
        print('softmax_out_avg loaded, shape: ', softmax_out_avg.shape)

    else:
        # Building model
        model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=args.dp, num_classes=10).to(device)

        # Training
        softmax_out_avg = np.zeros([len(train_dataset), 10])
        softmax_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        for epoch in range(1, args.epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)

        softmax_out_avg /= args.epochs
        np.save('data/CIFAR10/label_noisy/softmax_out_avg.npy', softmax_out_avg)

    if args.gen:
        print('Generating noisy labels according to softmax_out_avg...')
        label = np.array(train_dataset.targets)
        label_noisy_cand, label_noisy_prob = [], []
        for i in range(len(label)):
            pred = softmax_out_avg[i,:].copy()
            pred[label[i]] = -1
            label_noisy_cand.append(np.argmax(pred))
            label_noisy_prob.append(np.max(pred))
            
        label_noisy = label.copy()
        index = np.argsort(label_noisy_prob)[-int(args.noise_rate*len(label)):]
        label_noisy[index] = np.array(label_noisy_cand)[index]

        save_pth = os.path.join('./data/CIFAR10/label_noisy', 'dependent'+str(args.noise_rate)+'.csv')
        pd.DataFrame.from_dict({'label':label,'label_noisy':label_noisy}).to_csv(save_pth, index=False)
        print('Noisy label data saved to ',save_pth)

     
if __name__ == '__main__':
    main()
