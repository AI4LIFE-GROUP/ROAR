from sklearn.linear_model import LogisticRegression
import torch
from sklearn import metrics
import torch.nn as nn
import numpy as np 
from sklearn.svm import LinearSVC
from recourse_utils import DummyScaler

class Model():
	def __init__(self):
		pass

	#redo
	def metrics(self, X, y):
		acc = np.mean(self.predict(X)==y)

		pred = self.predict_proba(X)[:,1]
		fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)

		return acc, auc


class LR(Model):
	def __init__(self):
		super(LR, self).__init__()

	def train(self,X,y):
		sklearn_model = LogisticRegression().fit(X, y)
		self.sklearn_model = sklearn_model

		self.W = torch.from_numpy(sklearn_model.coef_[0]).float()
		self.W0 = torch.tensor(sklearn_model.intercept_[0]).float()

	def torch_model(self, x):
		return torch.nn.Sigmoid()(torch.matmul(self.W,x)+self.W0)

	def predict(self, x):
		return self.sklearn_model.predict(x)

	def predict_proba(self, x):
		return self.sklearn_model.predict_proba(x)


class SVM(Model):
	def __init__(self):
		super(SVM, self).__init__()

	def train(self,X,y):
		sklearn_model = LinearSVC().fit(X, y)
		self.sklearn_model = sklearn_model
		self.n_features = X.shape[1]
		self.W = torch.from_numpy(sklearn_model.coef_[0]).float()
		self.W0 = torch.tensor(sklearn_model.intercept_[0]).float()
		self.platt_scaling(self.sklearn_model.decision_function(X), y)

	def platt_scaling(self, X_train, y_train):
		self.ps = LogisticRegression().fit(X_train.reshape(-1, 1), y_train)
		self.pW = torch.tensor(self.ps.coef_).float()
		self.pW0 = torch.tensor(self.ps.intercept_).float()

	def torch_model(self, X):
		dec_fn = torch.matmul(X, self.W.T)+self.W0
		ps = torch.nn.Sigmoid()(torch.matmul(self.pW, dec_fn.unsqueeze(0))+self.pW0)
		return ps[0] 

	def predict(self, x):
		return (self.predict_proba(x)[:,1]>0.5).astype(int)

	def predict_proba(self, x):
		return self.ps.predict_proba(self.sklearn_model.decision_function(x).reshape(-1,1))


class NN(Model):
	def __init__(self, num_feat):
		torch.manual_seed(0)
		super(NN, self).__init__()
		self.net = nn.Sequential(
		  nn.Linear(num_feat, 50),
		  nn.ReLU(),
		  nn.Linear(50, 100),
		  nn.ReLU(),
		  nn.Linear(100, 200),
		  nn.ReLU(),
		  nn.Linear(200, 1),
		  nn.Sigmoid()
		  )
	
	def torch_model(self,x):
		return self.net(x)[0]

	def train(self, X_train, y_train):
		torch.manual_seed(0)
		X_train = torch.from_numpy(X_train).float()
		y_train = torch.from_numpy(y_train).float()

		criterion = nn.BCELoss()
		optimizer = torch.optim.Adam(self.net.parameters())

		# Train model
		epochs = 100
		for ep in range(epochs):
			self.net.train()
			
			optimizer.zero_grad()

			# Forward pass
			y_pred = self.net(X_train)

			# Compute Loss
			loss = criterion(y_pred[:,0], y_train)

			#print('Epoch {}: train loss: {}'.format(ep, loss.item()))

			# Backward pass
			loss.backward()
			
			optimizer.step()

	def predict_proba(self, X):
		X = torch.from_numpy(np.array(X)).float()
		class1_probs = self.net(X).detach().numpy()
		class0_probs = 1-class1_probs
		return np.hstack((class0_probs,class1_probs))

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)









