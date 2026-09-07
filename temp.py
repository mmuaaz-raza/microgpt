import numpy as np

# mask = np.tril(np.ones((3)))
# mask[mask==0] = -np.inf 
# mask[mask==1] = 0 
# m1= np.array([[[1,2,3]]
#      ,[[1,2,3]]])
# m2= np.array([[[2],[2],[2]],
#      [[2],[2],[2]]])

# print((m1@m2).shape)
# fr = 10000**(np.arange(0, 10, 2)/10)
# print(fr)
# T = [2,3,4]
# Wemd = np.random.randn(20, 12) * (1 / (12)**0.5)
# E = np.stack([ Ei:= Wemd[ti] for ti in T])
rng = np.random.default_rng()
tensor = rng.integers(low=0, high=5, size=(2, 3, 2))
print(tensor)
trtensor = np.transpose(tensor,axes=(1,0,2)) 
print(trtensor)
mask = np.tril(np.ones((3)))

print(np.tril(np.ones((3))))

print("""First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
"""[:64])
# print(np.sum(arr,axis=1))
