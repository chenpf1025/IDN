import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from loss import nll_loss_soft, forward_loss

""" Training/testing models """
# normal training
def train(args, model, device, loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(loader.dataset)
    print('Epoch: {}/{}\nTraining loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs, train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# SEAL training
def train_soft(args, model, device, loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for data, target_soft, target in loader:
        data, target_soft, target = data.to(device), target_soft.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss_soft(output, target_soft)
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(loader.dataset)
    print('Epoch: {}/{}\nTraining loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs, train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# dac
def train_dac(args, model, device, loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target, epoch)
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # If the maximum is the last entry, then it means abstained
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(loader.dataset)
    print('Epoch: {}/{}\nTraining loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs, train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# Forward Correction
def train_forward(args, model, device, loader, optimizer, epoch, T):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = forward_loss(output, target, T)
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(loader.dataset)
    print('Epoch: {}/{}\nTraining loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs, train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# Co-teaching
def train_ct(args, model1, model2, device, loader, optimizer1, optimizer2, epoch, p_keep):
    model1.train(), model2.train()
    train_loss1, train_loss2 = 0, 0
    correct1, correct2 = 0, 0
    for data, target in loader:
        n_keep = round(p_keep*data.size(0))
        data, target = data.to(device), target.to(device)
        output1, output2 = model1(data), model2(data)
        loss1, loss2 = F.nll_loss(output1, target, reduction='none'), F.nll_loss(output2, target, reduction='none')

        # selecting #n_keep small loss instances
        _, index1 = torch.sort(loss1.detach())
        _, index2 = torch.sort(loss2.detach())
        index1, index2 = index1[:n_keep], index2[:n_keep]

        # taking a optimization step
        optimizer1.zero_grad()
        loss1[index2].mean().backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2[index1].mean().backward()
        optimizer2.step()

        train_loss1, train_loss2 = train_loss1+loss1.sum().item(), train_loss2+loss2.sum().item()
        pred1, pred2 = output1.argmax(dim=1, keepdim=True), output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct1, correct2 = correct1+pred1.eq(target.view_as(pred1)).sum().item(), correct2+pred2.eq(target.view_as(pred2)).sum().item()
    train_loss1, train_loss2 = train_loss1/len(loader.dataset), train_loss2/len(loader.dataset)
    print('Epoch: {}/{}\nModel1 Training. Training loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)\nModel2 Training. Training loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs,
        train_loss1, correct1, len(loader.dataset), 100. * correct1 / len(loader.dataset),
        train_loss2, correct2, len(loader.dataset), 100. * correct2 / len(loader.dataset)))

# normal testing
def test(args, model, device, loader, top5=False):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()

    test_loss /= len(loader.dataset)
    if top5:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset), correct_k, len(loader.dataset), 100. * correct_k / len(loader.dataset)))
    else:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# with a validation set
def val_test(args, model, device, val_loader, test_loader, best_val_acc, save_path):
    model.eval()

    # val
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss, val_acc = loss/len(val_loader.dataset), 100.*correct/len(val_loader.dataset)

    # test
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss, test_acc = loss/len(test_loader.dataset), 100.*correct/len(test_loader.dataset)

    if val_acc>best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)

    print('Val loss: {:.4f}, Testing loss: {:.4f}, Val accuracy: {:.2f}%, Best Val accuracy: {:.2f}%, Testing accuracy: {:.2f}%\n'.format(
        val_loss, test_loss, val_acc, best_val_acc, test_acc))

    return best_val_acc


# dac
def test_dac(args, model, device, loader, epoch, criterion, top5=False):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, epoch).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # If the maximum is the last entry, then it means abstained
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()
    test_loss /= len(loader.dataset)
    if top5:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset), correct_k, len(loader.dataset), 100. * correct_k / len(loader.dataset)))
    else:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

def test_ct(args, model1, model2, device, loader, top5=False):
    model1.eval(), model2.eval()
    test_loss1, test_loss2 = 0, 0
    correct1, correct2 = 0, 0
    correct1_k, correct2_k = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model1(data), model2(data)
            test_loss1, test_loss2 = test_loss1+F.nll_loss(output1, target, reduction='sum').item(), test_loss2+F.nll_loss(output2, target, reduction='sum').item() # sum up batch loss
            pred1, pred2 = output1.argmax(dim=1, keepdim=True), output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct1, correct2 = correct1+pred1.eq(target.view_as(pred1)).sum().item(), correct2+pred2.eq(target.view_as(pred2)).sum().item()
            if top5:
                _, pred1 = output1.topk(5, 1, True, True)
                correct1_k += pred1.eq(target.view(-1,1)).sum().item()
                _, pred2 = output2.topk(5, 1, True, True)
                correct2_k += pred2.eq(target.view(-1,1)).sum().item()

    test_loss1, test_loss2 = test_loss1/len(loader.dataset), test_loss2/len(loader.dataset)
    if top5:
        print('Model1 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\nModel2 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\n'.format(
            test_loss1, correct1, len(loader.dataset), 100. * correct1 / len(loader.dataset), correct1_k, len(loader.dataset), 100. * correct1_k / len(loader.dataset),
            test_loss2, correct2, len(loader.dataset), 100. * correct2 / len(loader.dataset), correct2_k, len(loader.dataset), 100. * correct2_k / len(loader.dataset)))
    else:
        print('Model1 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\nModel2 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss1, correct1, len(loader.dataset), 100. * correct1 / len(loader.dataset),
            test_loss2, correct2, len(loader.dataset), 100. * correct2 / len(loader.dataset)))


def compute_CSR(args, model, device, loader, alpha=0.25/255, beta=0.2/255, r=0.3/255, max_iter=10):
    """
    Refference: Langevin Adversarial Sample Search (LASS) algorithm in the paper 'A Closer Look at Memorization in Deep Networks' <https://arxiv.org/abs/1706.05394>.
    """
    model.train()

    count_cs = 0
    for x, _ in loader:

        x_hat = Variable(x.data, requires_grad=True).to(device)
        x = x.to(device)
        pred_on_x = model(x).argmax(dim=1, keepdim=False)

        for i in range(max_iter):
            # compute gradient on x_hat
            x_hat = Variable(x_hat.cpu().data, requires_grad=True).to(device)
            x_hat.retain_grad()
            output_on_x_hat = model(x_hat)
            cost = -F.nll_loss(output_on_x_hat, pred_on_x)

            model.zero_grad()
            if x_hat.grad is not None:
                print('fill 0 to grad')
                x_hat.grad.data.fill_(0)
            cost.backward()

            # take a step
            noise = np.random.normal()
            x_hat.grad.sign_()
            x_hat = x_hat - (alpha*x_hat.grad + beta*noise)

            # projec back to the box
            x_hat = torch.max(torch.min(x_hat, x+r), x-r)
            
            # check is adversial
            index_not_adv = pred_on_x.view(-1,1).eq(model(x_hat).argmax(dim=1, keepdim=True)).view(-1)
            num_not_adv = index_not_adv.sum().item()

            # record number of adversial samples
            count_cs = count_cs + (pred_on_x.size(0)-num_not_adv)
            # print('count_cs: {}, num_not_adv: {}'.format(count_cs, num_not_adv))
            if num_not_adv>0:
                x_hat = x_hat[index_not_adv]#.unsqueeze(1)
                x = x[index_not_adv]#.unsqueeze(1)
                pred_on_x = pred_on_x[index_not_adv]#.view(-1)
            else:
                break

    return count_cs/len(loader.dataset)
