import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as scio
import pdb

BATCH_SIZE = 64

data = scio.loadmat('house.mat')

X = data['trainX']
Y = data['trainY']
testX = data['testX']
testY = data['testY']

x = torch.from_numpy(X).float()
y = torch.from_numpy(Y).float()

torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)

loader = Data.DataLoader(dataset=torch_dataset,
						batch_size = BATCH_SIZE,
						shuffle = True,
						num_workers = 2)

class LinearRegression(nn.Module):
	def __init__(self):
		super(LinearRegression, self).__init__()
		self.linear = nn.Linear(14,1)

	def forward(self,x):
		out = self.linear(x)
		return out

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))

l_his = []

for epoch in range(10):
	for i, (batch_x, batch_y) in enumerate(loader):
		inputs = Variable(batch_x)
		labels = Variable(batch_y)

		# pdb.set_trace()
		y_pred = model(inputs)
		loss = criterion(y_pred,labels)
		print(epoch, i, loss.data[0])
		l_his.append(loss.data[0])
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

plt.figure()
plt.plot(l_his)
plt.show()
print l_his
		# print('Epoch: ', epoch, '| Step: ', i, ' | batch y: ', batch_y.numpy())
