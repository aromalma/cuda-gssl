from gssl import CudaGSSL
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
np.random.seed(2)

iris = load_iris()

X = np.array(iris.data)
y = np.array(iris.target)
for ii in range(0,len(y),20):
    y[ii:ii+19]=-1
# print(y==-1,np.sum(y==-1))


scaler = StandardScaler()
# transform data
X = scaler.fit_transform(X)

yy=np.array(y)

uniq=set(yy)-{-1}
# make one ot encoding
one_hot=np.eye(len(uniq)+1,dtype=np.float64)[yy][:,:-1]
one_hot=np.array(one_hot) 

# init
g=CudaGSSL(X)
# gen W
g.gen_w()
# return W matrix
g.get_w()
# label propagation

p=g.label_prop(one_hot)
#
print(g.get_wi())


res=np.argmax(p,axis=1)

print("Total:",len(y))
# print(y)
print("total_labelled:",np.sum(y!=-1))
print("Total Unlabelled:",np.sum(y==-1))
print("Accuracy :",np.mean(iris.target==res)*100,"%")

