import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import pdb
import math

def nll_loss_soft(input, target):
    return -torch.mean(torch.sum(target*input, dim=1))

def forward_loss(input, target, T):
	input = torch.log(torch.matmul(torch.exp(input), T))
	out = input[range(input.size(0)), target]
	return -torch.mean(out)

"""
Existing robust learning method: Combating Label Noise in Deep Learning Using Abstention (https://github.com/thulas/dac-label-noise)
Loss calculation and alpha-ramping are rolled into one function. This is invoked after every iteration
"""
#for numerical stability
epsilon = 1e-7
class dac_loss(_Loss):
	def __init__(self, model, learn_epochs, total_epochs, use_cuda=False, cuda_device=None, 
			alpha_final=1.0, alpha_init_factor=64.):
		print("using dac loss function\n")
		super(dac_loss, self).__init__()
		self.model = model
		self.learn_epochs = learn_epochs
		self.total_epochs = total_epochs
		self.alpha_final = alpha_final
		self.alpha_init_factor = alpha_init_factor
		self.use_cuda = use_cuda
		self.cuda_device = cuda_device
		self.alpha_var = None
		self.alpha_thresh_ewma = None   #exponentially weighted moving average for alpha_thresh
		self.alpha_thresh = None #instantaneous alpha_thresh
		self.ewma_mu = 0.05 #mu parameter for EWMA; 
		self.curr_alpha_factor  = None #for alpha initiliazation
		self.alpha_inc = None #linear increase factor of alpha during abstention phase
		self.alpha_set_epoch = None


	def __call__(self, input_batch, target_batch, epoch):
		if epoch <= self.learn_epochs or not self.model.training:
			#pdb.set_trace()
			loss =  F.cross_entropy(input_batch, target_batch, reduction='none')
			#return loss.mean()
			if self.model.training:
				h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduction='none')
				p_out = torch.exp(F.log_softmax(input_batch,dim=1))
				p_out_abstain = p_out[:,-1]
				#pdb.set_trace()

				#update instantaneous alpha_thresh
				self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)
				#update alpha_thresh_ewma 
				if self.alpha_thresh_ewma is None:
					self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
				else:
					self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
						(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)
			return loss.mean()

		else:
			#calculate cross entropy only over true classes
			h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduce=False)
			p_out = torch.exp(F.log_softmax(input_batch,dim=1))
			#probabilities of abstention  class
			p_out_abstain = p_out[:,-1]

			if self.use_cuda:
				p_out_abstain = torch.min(p_out_abstain,
					Variable(torch.Tensor([1. - epsilon])).cuda(self.cuda_device))
			else:
				p_out_abstain = torch.min(p_out_abstain,
					Variable(torch.Tensor([1. - epsilon])))

			#update instantaneous alpha_thresh
			self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)

			try:
	    		#update alpha_thresh_ewma
				if self.alpha_thresh_ewma is None:
					self.alpha_thresh_ewma = self.alpha_thresh
				else:
					self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
						(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)


				if self.alpha_var is None:
					self.alpha_var = 	Variable(self.alpha_thresh_ewma.data /self.alpha_init_factor)
					self.alpha_inc =  (self.alpha_final - self.alpha_var.data)/(self.total_epochs - epoch)
					self.alpha_set_epoch = epoch

				else:		
					# we only update alpha every epoch
					if epoch > self.alpha_set_epoch: 
						self.alpha_var = Variable(self.alpha_var.data + self.alpha_inc)
						self.alpha_set_epoch = epoch

				loss = (1. - p_out_abstain)*h_c - \
		    		self.alpha_var*torch.log(1. - p_out_abstain)
				return loss.mean()
			except RuntimeError as e:
				print(e)