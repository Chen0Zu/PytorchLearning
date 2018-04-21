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

data = scio.loadmat('mnist_multi.mat')
train = data['train']
X = torch.from_numpy(train['X'][0][0].T).float()
y = torch.from_numpy(train['y'][0][0].T).long()
y = y.view(-1)

testX = torch.from_numpy(data['test']['X'][0][0].T).float()
testy = torch.from_numpy(data['test']['y'][0][0].T).long().view(-1)

BATCH_SIZE = 32
torch_dataset = Data.TensorDataset(data_tensor=X, target_tensor=y)

loader = Data.DataLoader(
			dataset = torch_dataset,
			batch_size = BATCH_SIZE,
			shuffle=True,
			num_workers = 2,
			)

class MLP(nn.Module):
	def __init__(self):
		super(MLP,self).__init__()
		self.fc1 = nn.Linear(28*28,256)
		self.fc2 = nn.Linear(256, 10)

	def forward(self,x):
		x = self.fc1(x)
		x = self.fc2(x)
		return x

# pdb.set_trace()
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01,betas=(0.9,0.99))

l_his = []
for epoch in range(100):
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
dump,idx = pred_y.max(1)
acc_train = torch.mean((idx==y).float())

pred_y = model(Variable(testX))
pred_y = pred_y.data
dump,idx = pred_y.max(1)
acc_test = torch.mean((idx==testy).float())

print("Training accuracy: %f\n" % acc_train)
print("Testing accuracy: %f\n" % acc_test)
# plt.figure()
# plt.plot(l_his)
# plt.show()
# pdb.set_trace()
