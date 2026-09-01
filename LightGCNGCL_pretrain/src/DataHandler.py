import pickle
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
# from Params import args
import scipy.sparse as sp
# from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

class DataHandler:
	def loadfeature(self):
		with open(r'C:\Users\24737\PycharmProjects\AdaptiveGCL-ma\Datasets\api\nm_tensor.pt', 'rb') as fs:
			self.uEmbeds = torch.load(fs)
			# uEmbeds = [np.float64(1.0), np.float64(2.0), np.float64(3.0)]

			# self.uEmbeds = np.array(uEmbeds)

			# 将列表中的值转换为一个包含单一值的 NumPy 数组
		with open(r'C:\Users\24737\PycharmProjects\AdaptiveGCL-ma\Datasets\api\na_tensor.pt', 'rb') as fs:
			self.iEmbeds = torch.load(fs)
			# self.iEmbeds = np.array(iEmbeds)
			self.aEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)
	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])