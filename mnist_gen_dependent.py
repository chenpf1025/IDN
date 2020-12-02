
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from utils import get_softmax_out
from ops import train, test
from networks.cnn_mnist import MNIST_CNN
 
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)') # On clean data, 20 is sufficiently large to achiece 100% training accuracy.
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate (default: 0.0)')
    parser.add_argument('--load', action='store_true', default=False, help='Load existing averaged softmax')
    parser.add_argument('--gen', action='store_true', default=False, help='Generate noisy labels')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = './data'
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #0.1307, 0.3081 are the mean and std of mnist
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if args.load:
        softmax_out_avg = np.load('data/MNIST/label_noisy/softmax_out_avg.npy')
        print('softmax_out_avg loaded, shape: ', softmax_out_avg.shape)

    else:
        # Building model
        model = MNIST_CNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Training
        softmax_out_avg = np.zeros([len(train_dataset), 10])
        softmax_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)

        softmax_out_avg /= args.epochs
        np.save('data/MNIST/label_noisy/softmax_out_avg.npy', softmax_out_avg)

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

        save_pth = os.path.join('./data/MNIST/label_noisy', 'dependent'+str(args.noise_rate)+'.csv')
        pd.DataFrame.from_dict({'label':label,'label_noisy':label_noisy}).to_csv(save_pth, index=False)
        print('Noisy label data saved to ',save_pth)
     
if __name__ == '__main__':
    main()