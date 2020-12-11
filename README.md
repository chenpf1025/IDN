# Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise.
This is the official repository for the paper [Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise](https://arxiv.org/abs/2012.05458). In this paper, one of the contributions is to provide rigorous motivations for studying instance-dependent label noise.
```
@article{chen2020beyond,
  title={Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise.},
  author={Chen, Pengfei and Ye, Junjie and Chen, Guangyong and Zhao, Jingwei and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2012.05458},
  year={2020}
}
```

## 0. Requirements
* python 3.6+
* torch 1.2+

## 1. Instance-Dependent Noise (IDN)
### 1.1. Noisy labels used in this paper
In our experiments, we generated noisy labels of IDN for MNIST and CIFAR-10. Here we release the related files.
```
data/CIFAR10/label_noisy/dependent0.1.csv
data/CIFAR10/label_noisy/dependent0.2.csv
data/CIFAR10/label_noisy/dependent0.3.csv
data/CIFAR10/label_noisy/dependent0.4.csv
data/MNIST/label_noisy/dependent0.1.csv
data/MNIST/label_noisy/dependent0.2.csv
data/MNIST/label_noisy/dependent0.3.csv
data/MNIST/label_noisy/dependent0.4.csv
```

If you are developing novel methods, you are encouraged to use these files for a fair comparison with the results reported in our paper. The index in the .csv file is consistent with the default dataset in torchvision. For example, to get a CIFAR-10 dataset with 40% IDN, you can use the following scripts in you code.
```
from torchvision import datasets
train_dataset_noisy = datasets.CIFAR10(root, train=True, download=True, transform=transform)
targets_noisy = list(pd.read_csv('./data/CIFAR10/label_noisy/dependent0.4.csv')['label_noisy'].values.astype(int))
train_dataset_noisy.targets = targets_noisy
```
To get a MNIST dataset with 40% IDN, you can use the following scripts in you code.
```
from torchvision import datasets
train_dataset_noisy = datasets.MNIST(root, train=True, download=True, transform=transform)
targets_noisy = torch.Tensor(pd.read_csv('./data/MNIST/label_noisy/dependent0.4.csv')['label_noisy'].values.astype(int))
train_dataset_noisy.targets = targets_noisy
```

### 1.2. Synthetizing IDN
If you prefer to synthetize IDN, e.g., to synthetize 45% IDN for CIFAR-10, you can use the following commands.
```
python cifar10_gen_dependent.py --noise_rate 0.45 --gen
```
The command will train a model on clean CIFAR-10, yield the average of softmax output, and then synthetize IDN. After you running the command for the first time, the averaged softmax output is saved and you can directly generate IDN of any other ratio by loading it, e.g.,
```
python cifar10_gen_dependent.py --noise_rate 0.35 --gen --load
```

If you need to write a script to synthetize IDN for a new dataset, you can refer to the file *mnist_gen_dependent.py* and *cifar10_gen_dependent.py*.


## 2. Combating IDN using SEAL
### 2.1. MNIST
For SEAL, we use 10 iterations. We can run the commands one-by-one as follows.
```
python train_mnist.py --noise_rate 0.2 --SEAL 0 --save
python train_mnist.py --noise_rate 0.2 --SEAL 1 --save
...
python train_mnist.py --noise_rate 0.2 --SEAL 10 --save
```
The initial iteration is equivalent to training using the cross-entropy (CE) loss. To run experiments on different noise fractions, 
we can choose --noise_rate in {0.1,0.2,0.3,0.4}.

### 2.2. CIFAR-10
For SEAL, we use 3 iterations. We can run the commands one-by-one as follows.
```
python train_cifar10.py --noise_rate 0.2 --SEAL 0 --save
python train_cifar10.py --noise_rate 0.2 --SEAL 1 --save
python train_cifar10.py --noise_rate 0.2 --SEAL 2 --save
python train_cifar10.py --noise_rate 0.2 --SEAL 3 --save
```
The initial iteration is equivalent to training using the cross-entropy (CE) loss. To run experiments on different noise fractions, 
we can choose --noise_rate in {0.1,0.2,0.3,0.4}.

### 2.3. Clothing1M
By default, the training requirements 4 GPUs.
For SEAL, we use 3 iterations. We can run the commands one-by-one as follows.
```
python train_clothing.py --SEAL 0 --save
python train_clothing.py --SEAL 1 --save
python train_clothing.py --SEAL 2 --save
python train_clothing.py --SEAL 3 --save
```
The initial iteration is equivalent to training using the cross-entropy (CE) loss

To run SEAL on top of DMI, we first use the official implementation of DMI to obtained a model, then use the following commands one-by-one.
```
python train_clothing_dmi.py --SEAL 1 --save
python train_clothing_dmi.py --SEAL 2 --save
python train_clothing_dmi.py --SEAL 3 --save
```
