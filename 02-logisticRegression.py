import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np 
import scipy.io as scio
import pdb


# pdb.set_trace()
plot_sigmoid = False
if plot_sigmoid:
	x = torch.from_numpy(np.linspace(-10,10,100)).float().view(-1,1)
	y = F.sigmoid(Variable(x))

	plt.figure()
	plt.plot(x.numpy(),y.data.numpy())
	plt.show()

data = scio.loadmat('mnist.mat')
train = data['train']
X = torch.from_numpy(train['X'][0][0].T).float()
y = torch.from_numpy(train['y'][0][0].T).float()

testX = torch.from_numpy(data['test']['X'][0][0].T).float()
testy = torch.from_numpy(data['test']['y'][0][0].T).float()

BATCH_SIZE = 32
torch_dataset = Data.TensorDataset(data_tensor=X, target_tensor=y)

loader = Data.DataLoader(
			dataset = torch_dataset,
			batch_size = BATCH_SIZE,
			shuffle=True,
			num_workers = 2,
			)

class logisticRegression(nn.Module):
	def __init__(self):
		super(logisticRegression,self).__init__()
		self.linear = nn.Linear(28*28,1)

	def forward(self,x):
		y = F.sigmoid(self.linear(x))
		return y

model = logisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

l_his = []
for epoch in range(10):
	for step, (batch_x, batch_y) in enumerate(loader):

		inputs = Variable(batch_x)
		labels = Variable(batch_y)

		out = model(inputs)
		loss = criterion(out,labels)
		print(epoch, step, loss.data[0])
		l_his.append(loss.data[0])
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

pred_y = model(Variable(X))
pred_y = pred_y.data
pred_y[pred_y>0.5] = 1
pred_y[pred_y<=0.5] = 0
acc_train = torch.mean((pred_y==y).float())

pred_y = model(Variable(testX))
pred_y = pred_y.data
pred_y[pred_y>0.5] = 1
pred_y[pred_y<=0.5] = 0
acc_test = torch.mean((pred_y==testy).float())
# plt.figure()
# plt.plot(l_his)
# plt.show()
pdb.set_trace()
