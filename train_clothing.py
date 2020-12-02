import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from utils import get_softmax_out
from ops import train, train_soft, test, val_test
from dataset import Clothing1M, Clothing1M_soft
from networks.resnet import resnet50

   
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Clothing1M')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 120)')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--save', action='store_true', default=False, help='For saving softmax_out_avg')
    parser.add_argument('--SEAL', type=int, default=0, help='Phase of self-evolution')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = './data/Clothing1M'
    num_classes = 14
    kwargs = {'num_workers': 32, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform_train = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
    train_dataset = Clothing1M(root, mode='train', transform=transform_train)
    val_dataset = Clothing1M(root, mode='val', transform=transform_test)
    test_dataset = Clothing1M(root, mode='test', transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    softmax_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    def learning_rate(lr_init, epoch):
        optim_factor = 0
        if(epoch > 5):
            optim_factor = 1
        return lr_init*math.pow(0.1, optim_factor)

    def load_pretrain(num_classes, device):
        model_pre = resnet50(num_classes=1000, pretrained=True) # imagenet pretrained, numclasses=1000
        if num_classes==1000:
            return model_pre.to(device)

        else:
            model = resnet50(num_classes=num_classes, pretrained=False)
            params_pre = model_pre.state_dict().copy()
            params = model.state_dict()
            for i in params_pre:
                if not i.startswith('fc'):
                    params[i] = params_pre[i]
            model.load_state_dict(params)
            return model.to(device)

    # results
    results_root = os.path.join('results', 'clothing')
    if not os.path.isdir(results_root):
        os.makedirs(results_root)

    """ Test model """
    if args.SEAL==-1:
        model = resnet50().to(device)
        model.load_state_dict(torch.load(os.path.join(results_root, 'seed0_clothing_normal.pt')))
        test(args, model, device, test_loader)


    """ Get softmax_out_avg - normal training on noisy labels """
    if args.SEAL==0:
        # Building model
        model = load_pretrain(num_classes, device)
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

        # Training
        best_val_acc = 0
        save_path = os.path.join(results_root, 'seed'+str(args.seed)+'_clothing_normal.pt')
        softmax_out = []
        for epoch in range(1, args.epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=1e-3)
            train(args, model, device, train_loader, optimizer, epoch)
            best_val_acc = val_test(args, model, device, val_loader, test_loader, best_val_acc, save_path)
            softmax_out.append(get_softmax_out(model, softmax_loader, device))

        if args.save:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_normal.npy')
            softmax_out = np.concatenate(softmax_out)
            np.save(softmax_root, softmax_out)
            print('new softmax_out saved to', softmax_root, ', shape: ', softmax_out.shape)


    """ Self Evolution - training on softmax_out_avg """
    if args.SEAL>=1:
        if args.SEAL==1:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_normal.npy')
            model_path = os.path.join(results_root, 'seed'+str(args.seed)+'_clothing_normal.pt')
        else:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_SEAL'+str(args.SEAL-1)+'.npy')
            model_path = os.path.join(results_root, 'seed'+str(args.seed)+'_clothing_SEAL'+str(args.SEAL-1)+'.pt')

        save_path = os.path.join(results_root, 'seed'+str(args.seed)+'_clothing_SEAL'+str(args.SEAL)+'.pt')   
            
        # Loading softmax_out_avg of last phase
        softmax_out_avg = np.load(softmax_root).reshape([-1, len(train_dataset), num_classes])
        softmax_out_avg = softmax_out_avg.mean(axis=0)
        print('softmax_out_avg loaded from', softmax_root, ', shape: ', softmax_out_avg.shape)

        # Dataset with soft targets
        train_dataset_soft = Clothing1M_soft(root, targets_soft=torch.Tensor(softmax_out_avg.copy()), mode='train', transform=transform_train)
        train_loader_soft = torch.utils.data.DataLoader(train_dataset_soft, batch_size=args.batch_size, shuffle=True, **kwargs)

        # Building model
        model = load_pretrain(num_classes, device)
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        model.load_state_dict(torch.load(model_path))
        print('Initialize the model using {}.'.format(model_path))

        # Training
        best_val_acc = 0
        softmax_out = []
        for epoch in range(1, args.epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=1e-3)
            train_soft(args, model, device, train_loader_soft, optimizer, epoch)
            best_val_acc = val_test(args, model, device, val_loader, test_loader, best_val_acc, save_path)
            softmax_out.append(get_softmax_out(model, softmax_loader, device))

        if args.save:
            softmax_root = os.path.join(results_root, 'seed'+str(args.seed)+'_softmax_out_SEAL'+str(args.SEAL)+'.npy')
            softmax_out = np.concatenate(softmax_out)
            np.save(softmax_root, softmax_out)
            print('new softmax_out saved to', softmax_root, ', shape: ', softmax_out.shape)
     
if __name__ == '__main__':
    main()