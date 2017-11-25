import numpy as np
from sklearn.decomposition import PCA
import scipy

C =np.mat([[1,2,3],[4,5,5],[7,8,9]])
print(np.multiply(np.sqrt(C),np.sqrt(C)))
print(np.linalg.inv(C))
print(C)
eig_val, eig_vec = np.linalg.eig(C)
print(eig_val)
eig_val = eig_val.real
print(eig_val)
print(eig_vec)
pca = PCA(n_components=2)
new_data = pca.fit_transform(C)
print(new_data)
data_id = np.mat([0,-1,-2,0,1,0,-1,0])
m = [0, 1]
data_id[:, m] = 10
data_id = np.delete(data_id, m, axis=1)
s = abs(data_id)
t = data_id/0
ss = np.isinf(t)
m = np.where(ss == True)[1]
print(len(data_id))
E = np.mat([-1.,1.,2.,-5.,7.,8.])
zero_v = np.where(E <= 0)[1]
sss = zero_v[0]
if len(zero_v) > 0:
    E[:, zero_v] = 1E-100
print(E)
print(E[0,0])

print(t[np.isnan(t)])
print(data_id.shape)
id_temp_index = np.where(data_id == 0)[0]
print(id_temp_index)
priors= len(id_temp_index)
print(priors)
l = np.mat([1,2,3,4])
print(len(l))
print(l.shape)
print(len(l.T))
print((l.T).shape)
m = np.mat([[1,3],[2,6],[3,9],[4,12]])
n = np.array([[10,1], [20,2], [30,3], [40,4]])
print(np.where(n == 3)[0])
print(m/2)
print(m/n)
print(np.exp(-0.5 * m))
print(m)
print(m.T)
print(m.I)
print(m*m.I)
'''
priors = [1,2,3,4]
print(priors)
priors = priors / np.sum(priors)
print(priors)
m = np.mat([[1.,2.,3.,4.],[2.,3.,4.,5.],[3.,2.,5.,6.]])
print(m)
d = [0,2]
print(m[:,d])
a = m[:,d]
t = (np.tile(a, (1, 2))).T
print(t)
s = np.cov(t)
print(np.cov(t.T))
print(s[0][0])
X = np.mat([[1, 5, 6], [4, 3, 9], [4, 2, 9], [4, 7, 2]])
print(X)
print(np.cov(X.T))
'''