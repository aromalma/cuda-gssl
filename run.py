import numpy as np 
a= np.random.randint(5,size=(3,3),)
# a=np.ones((3,3))
a=a.astype(np.float32)
print("Input:\n",a)
from gssl import CudaGSSL 
g=CudaGSSL(a)
g.gen_w()
print("")
print("weight matrix:\n",g.get_w())
