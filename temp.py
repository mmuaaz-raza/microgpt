import numpy as np

# arr = np.array([[2,3],[4,5],[8,7]])
positions = np.arange(8)[:,np.newaxis]
print(positions)
divisor =  10000**(np.arange(0,128,2)/128)
print(divisor)
# print(np.sum(arr,axis=1))
