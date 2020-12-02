import torch
import torch.nn as nn
import torch.nn.functional as F
 
""" MNIST CNN """
class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10, use_log_softmax=True):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.use_log_softmax = use_log_softmax

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.use_log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x