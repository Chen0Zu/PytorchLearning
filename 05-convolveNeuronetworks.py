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

data = scio.loadmat('cnnMnist.mat')
images = data['images'][:,np.newaxis,:,:]
testImages = data['testImages'][:,np.newaxis,:,:]
labels = data['labels']
labels[labels == 10] = 0
testLabels = data['testLabels']
testLabels[testLabels == 10] = 0
# pdb.set_trace()
X = torch.from_numpy(images).float()
y = torch.from_numpy(labels).long().view(-1)

testX = torch.from_numpy(testImages).float()
testy = torch.from_numpy(testLabels).long().view(-1)

BATCH_SIZE = 256
torch_dataset = Data.TensorDataset(data_tensor=X, target_tensor=y)

loader = Data.DataLoader(
			dataset = torch_dataset,
			batch_size = BATCH_SIZE,
			shuffle = True,
			num_workers = 2,
			)

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv = nn.Conv2d(1, 20, 9)
		self.avep = nn.AvgPool2d(2)
		self.fc = nn.Linear(2000,10)

	def forward(self,x):
		# pdb.set_trace()
		x = self.conv(x)
		
		x = F.sigmoid(x)
		x = self.avep(x)
		x = x.view(x.size(0),-1)
		x = self.fc(x)

		return x

# pdb.set_trace()
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01,betas=(0.9,0.99))

l_his = []
for epoch in range(3):
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
# pdb.set_trace()
acc_train = torch.mean((idx==y).float())

pred_y = model(Variable(testX))
pred_y = pred_y.data
dump,idx = pred_y.max(1)
acc_test = torch.mean((idx==testy).float())

print("Training accuracy: %f\n" % acc_train)
print("Testing accuracy: %f\n" % acc_test)