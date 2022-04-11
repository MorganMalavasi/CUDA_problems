import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
import time
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
culinalg.init()

'''
A = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9])).astype(np.float64)
B = np.array(([7, 8, 1])).astype(np.float64)
'''
M = 14000

A = np.arange(M*M).reshape(M,M).astype(np.float64)
B = np.arange(M).reshape(M).astype(np.float64)

start = time.time()
C_CPU = np.dot(A,B)
end = time.time()
print('%2.2f sec' % (end-start))

start = time.time()
A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)
C_GPU = culinalg.dot(A_gpu, B_gpu)
C_GPU = C_GPU.get()
end = time.time()
print('%2.2f sec' % (end-start))

# print(np.dot(A, B))
# print(C_gpu)

print((C_CPU==C_GPU).all())
    