import numpy as np

mask = np.tril(np.ones((3)))
mask[mask==0] = -np.inf 
mask[mask==1] = 0 
print(mask,np.ones((3)))
# print(np.sum(arr,axis=1))
