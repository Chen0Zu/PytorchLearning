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
class RNN(nn.Module):
	def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
		super(RNN,self).__init__()
		self.rnn = nn.RNN(in_feature, hidden_feature,num_layers)
		self.classifier = nn.Linear(hidden_feature, num_class)

	def forward(self,x):
		x = x.squeeze()
		x = x.permute(2,0,1)
		out,_=self.rnn(x)
		out = out[-1,:,:]
		out = self.classifier(out)
		return out

model = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), 1e-1)

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