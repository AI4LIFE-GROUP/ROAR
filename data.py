
import ipdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from carla import Data 
from abc import abstractmethod

class Dataset():
	def __init__(self, fold):
		#fold \in {0,1,2,3,4}
		self.fold = fold

	def load_data(self,fname, sep=","):
		df = pd.read_csv(fname, sep=sep)
		df = df.sample(frac=1,random_state=1)    
		return df 

	def get_feat_types(self, df):
		cat_feat = []
		num_feat = []
		for key in list(df):
			if df[key].dtype==object:
				cat_feat.append(key)
			elif len(set(df[key]))>2:
				num_feat.append(key)
		return cat_feat,num_feat

	def scale_num_feats(self, df1, df2, num_feat):
		#scale numerical features
		for key in num_feat:
			scaler = StandardScaler()
			df1[key] = scaler.fit_transform(df1[key].values.reshape(-1,1))
			df2[key] = scaler.transform(df2[key].values.reshape(-1,1))
		return df1, df2
	
	def split_data(self, X, y):
		x_chunks = []
		y_chunks = []
		for i in range(5):
			start = int(i/5*len(X))
			end = int((i+1)/5*len(X))
			x_chunks.append(X.iloc[start:end])
			y_chunks.append(y.iloc[start:end])

		X_test, y_test = x_chunks.pop(self.fold), y_chunks.pop(self.fold)
		X_train, y_train = pd.concat(x_chunks), pd.concat(y_chunks)
		return X_train, y_train, X_test, y_test


class CorrectionShift(Dataset):
	def __init__(self, seed):
		super(CorrectionShift, self).__init__(seed)

	def get_data(self, fname1, fname2):
		df1 = self.load_data(fname1)
		df2 = self.load_data(fname2)

		#Using a reduced feature space due to causal baseline's SCM
		num_feat = ["duration", "amount", "age"]
		cat_feat = ["personal_status_sex"]
		target = "credit_risk"

		df1 = df1.drop(columns=[c for c in list(df1) if c not in num_feat+cat_feat+[target]])
		df2 = df2.drop(columns=[c for c in list(df2) if c not in num_feat+cat_feat+[target]])

		#Scale numerical features
		df1, df2 = self.scale_num_feats(df1, df2, num_feat)

		#One-hot encode categorical features
		df1 = pd.get_dummies(df1, columns=cat_feat)
		df2 = pd.get_dummies(df2, columns=cat_feat)

		X1, y1 = df1.drop(columns=[target]), df1[target]
		X2, y2 = df2.drop(columns=[target]), df2[target]
		
		data1 = self.split_data(X1, y1)
		data2 = self.split_data(X2, y2)

		return data1, data2

# class CorrectionShiftCarla(Data):
# 	def __init__(self, seed, fname1, fname2):
# 		d1, d2 = CorrectionShift(seed).get_data(fname1, fname2)
# 		#Passing D1 training set to recourse algorithm
# 		X1_train, y1_train, X1_test, y1_test = d1
# 		X1_train["credit_risk"] = y1_train.values
# 		self._dataset = X1_train
	
# 	@property
# 	def categoricals(self):
# 		return [c for c in list(self._dataset) if "sex" in c]
	
# 	@property
# 	def continous(self):
# 		return ["duration", "amount", "age"]

# 	@property
# 	def immutables(self):
# 		return [c for c in list(self._dataset) if "sex" in c]

# 	@property
# 	def target(self):
# 		return "credit_risk"

# 	@property
# 	def raw(self):
# 		return self._dataset



class TemporalShift(Dataset):
	def __init__(self, seed):
		super(TemporalShift, self).__init__(seed)

	def get_data(self, fname):
		df = self.load_data(fname)
		df = df.fillna(-1)

		#Define target variable
		df["NoDefault"] = 1-df["Default"].values

		#Drop unique identifiers, constants, feature perfectly correlated
		#with outcome, and categorical variables that blow up the 
		#feature space
		df = df.drop(columns=["Selected","State","Name", "BalanceGross", "LowDoc","BankState",
			"LoanNr_ChkDgt","MIS_Status","Default", "Bank", "City"])

		cat_feat,num_feat = self.get_feat_types(df)

		#One-hot encode categorical features
		df = pd.get_dummies(df, columns=cat_feat)

		#Get df1 and df2
		df1 = df[df["ApprovalFY"]<2006]
		df2 = df

		#Scale numerical features
		df1, df2 = self.scale_num_feats(df1, df2, num_feat)

		X1, y1 = df1.drop(columns=["NoDefault"]), df1["NoDefault"]
		X2, y2 = df2.drop(columns=["NoDefault"]), df2["NoDefault"]
		
		data1 = self.split_data(X1, y1)
		data2 = self.split_data(X2, y2)

		return data1, data2


# class TemporalShiftCarla(Data):
# 	def __init__(self, seed, fname):
# 		self.data = TemporalShift(seed)
# 		d1, d2 = self.data.get_data(fname)
# 		#Passing D1 training set to recourse algorithm
# 		X1_train, y1_train, X1_test, y1_test = d1
# 		X1_train["NoDefault"] = y1_train.values
# 		self._dataset = X1_train
	
# 	@property
# 	def categoricals(self):
# 		cat_feat,num_feat = self.data.get_feat_types(self._dataset)
# 		cat_feat = [c for c in list(self._dataset) if c not in num_feat]
# 		return [c for c in cat_feat if self.target not in c]
	
# 	@property
# 	def continous(self):
# 		cat_feat,num_feat = self.data.get_feat_types(self._dataset)
# 		return num_feat

# 	@property
# 	def immutables(self):
# 		return []

# 	@property
# 	def target(self):
# 		return "NoDefault"

# 	@property
# 	def raw(self):
# 		return self._dataset


class GeospatialShift(Dataset):
	def __init__(self, seed):
		super(GeospatialShift, self).__init__(seed)

	def get_data(self, fname, sep):
		df = self.load_data(fname, sep)

		#Define target variable
		df["Outcome"] = (df["G3"]<10).astype(int)

		#Drop variables highly correlated with target
		df = df.drop(columns=["G1","G2","G3"])

		cat_feat,num_feat = self.get_feat_types(df)

		#One-hot encode categorical features
		df = pd.get_dummies(df, columns=cat_feat)

		#Get df1 and df2
		df1 = df[df["school_GP"]==1]
		df2 = df

		#Scale numerical features
		df1, df2 = self.scale_num_feats(df1, df2, num_feat)

		X1, y1 = df1.drop(columns=["Outcome","school_GP","school_MS"]), df1["Outcome"]
		X2, y2 = df2.drop(columns=["Outcome","school_GP","school_MS"]), df2["Outcome"]
		
		data1 = self.split_data(X1, y1)
		data2 = self.split_data(X2, y2)

		return data1, data2


# class GeospatialShiftCarla(Data):
# 	def __init__(self, seed, fname, sep):
# 		self.data = GeospatialShift(seed)
# 		d1, d2 = self.data.get_data(fname, sep)
# 		#Passing D1 training set to recourse algorithm
# 		X1_train, y1_train, X1_test, y1_test = d1
# 		X1_train["Outcome"] = y1_train.values
# 		self._dataset = X1_train
	
# 	@property
# 	def categoricals(self):
# 		cat_feat,num_feat = self.data.get_feat_types(self._dataset)
# 		cat_feat = [c for c in list(self._dataset) if c not in num_feat]
# 		return [c for c in cat_feat if self.target not in c]
	
# 	@property
# 	def continous(self):
# 		cat_feat,num_feat = self.data.get_feat_types(self._dataset)
# 		return num_feat

# 	@property
# 	def immutables(self):
# 		return []

# 	@property
# 	def target(self):
# 		return "Outcome"

# 	@property
# 	def raw(self):
# 		return self._dataset


class SimulatedData(Dataset):
	def __init__(self, seed):
		self.c0_means = -2*np.ones(2)
		self.c1_means = 2*np.ones(2)
		self.c0_cov = 0.5*np.eye(2)
		self.c1_cov = 0.5*np.eye(2)
		super(SimulatedData, self).__init__(seed)
		
	def get_data(self, num_samples=1000):
		np.random.seed(1)
		
		X0 = np.random.multivariate_normal(self.c0_means,self.c0_cov,int(num_samples/2))
		X1 = np.random.multivariate_normal(self.c1_means,self.c1_cov,int(num_samples/2))
		X = np.vstack(np.array([X0,X1]))
		y = np.array([0]*int(num_samples/2)+[1]*int(num_samples/2))

		indices = np.arange(num_samples)
		np.random.shuffle(indices)
		X = X[indices]
		y = y[indices]
		data = pd.DataFrame({"X0":X[:,0],"X1":X[:,1], "y":y})
		X, y = data.drop(columns=["y"]), data["y"]

		X_train, y_train, X_test, y_test = self.split_data(X, y)
		#return X_train.values, X_test.values, y_train.values, y_test.values
		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.fold)
		return X_train, X_test, y_train, y_test

# class SimulatedDataCarla(Data):
# 	def __init__(self, seed, num_samples):
# 		self.data = SimulatedData(seed)
# 		X_train, X_test, y_train, y_test = self.data.get_data(num_samples)
# 		#Passing training set to recourse algorithm
# 		X_train["y"] = y_train.values
# 		self._dataset = X_train
	
# 	@property
# 	def categoricals(self):
# 		return []
	
# 	@property
# 	def continous(self):
# 		cat_feat,num_feat = self.data.get_feat_types(self._dataset)
# 		return ["X0", "X1"]

# 	@property
# 	def immutables(self):
# 		return []

# 	@property
# 	def target(self):
# 		return "y"

# 	@property
# 	def raw(self):
# 		return self._dataset


