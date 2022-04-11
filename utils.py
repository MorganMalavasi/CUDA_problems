
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import random
import math
from numba import cuda, types, float32, float64

'''
def checkResultsTable(array1, array2):
    # check sizes
    if array1.shape[0] != array2.shape[0] or array1.shape[1] != array2.shape[1]:
        return False

    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            
            # print("-> {0} - {1} = {2}".format(  abs(array1[i,j]), abs(array2[i,j]), abs(array1[i,j]) - abs(array2[i,j])) )
            
            if abs(abs(array1[i,j]) - abs(array2[i,j])) > 0.000000001 :
            # if abs(array1[i, j]) - abs(array2[i, j]) != 0:
                print(abs(array1[i, j]) - abs(array2[i, j]))
                return False
    return True

def distance(A, B):
    A_copy = np.copy(A)
    B_copy = np.copy(B)

    print("---------------(A_copy**2).sum(axis=1)[:,np.newaxis]")
    print((A_copy**2).sum(axis=1)[:,np.newaxis])
    print()
    
    print("---------------(B_copy**2).sum(axis=1)")
    print((B_copy**2).sum(axis=1))
    print()
    
    print("---------------(A_copy**2).sum(axis=1)[:,np.newaxis] + (B_copy**2).sum(axis=1)")
    print((A_copy**2).sum(axis=1)[:,np.newaxis] + (B_copy**2).sum(axis=1))
    print()

    print("---------------2*np.dot(A_copy,B_copy.T)")
    print(2*np.dot(A_copy,B_copy.T))
    print()
    
    print("---------------(A_copy**2).sum(axis=1)[:,np.newaxis] + (B_copy**2).sum(axis=1) - 2*np.dot(A_copy,B_copy.T)")
    print(((A_copy**2).sum(axis=1)[:,np.newaxis] + (B_copy**2).sum(axis=1)) - (2*np.dot(A_copy,B_copy.T)))
    print()


    dist = ((A_copy**2).sum(axis=1)[:,np.newaxis] + (B_copy**2).sum(axis=1) - 2*np.dot(A_copy,B_copy.T))
    return dist


a = np.arange(start = -4, stop = 8).reshape(3,4)
a = a * random.random()
print("---------------")
print(a)
print("---------------")


ini_array = distance(a, a)

for i in range(ini_array.shape[0]):
    for j in range(ini_array.shape[1]):
        if ini_array[i,j] > -0.00001 and ini_array[i,j] < 0.00001:
            ini_array[i,j] = 0.0

result = np.sqrt(ini_array)
b = np.where(np.isnan(result), 0, result)


print()
print(b)
print("|||||||||||||||")
u = euclidean_distances(a, a)
print(u)

print(checkResultsTable(b, u))



import numpy as np
from numba import cuda, types, float32

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    TPB = N

    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

TPB = 3
@cuda.jit

def fast_matmul1(A, B, C):

    """

    Perform matrix multiplication of C = A * B using CUDA shared memory.
    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella

    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)

    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0

        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]


        # Wait until all threads finish computing
        cuda.syncthreads()

    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

# This part is for initializing everything
M = 10
N = 5


#a = np.arange(M*N).reshape(M,N).astype(np.float32)
#b = np.arange(M*N).reshape(N,M).astype(np.float32)
a = np.ones(M*N).reshape(M,N).astype(np.float32)
b = np.ones(M*N).reshape(N,M).astype(np.float32)
c = np.zeros((M, M)).astype(np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

block_size = (N,N)
grid_size = (int(M/N),int(M/N))

fast_matmul1[grid_size,block_size](d_a, d_b, d_c)
c = d_c.copy_to_host()
print(c)


'''




TPB = 2


@cuda.jit
def kernel_computing6(theta, W, S, C):
    sA = cuda.shared.array(shape = (TPB), dtype = float64)
    sB = cuda.shared.array(shape = (TPB), dtype = float64)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= W.shape[0] or y >= W.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    sA[ty] = W[x, y]
    sB[ty] = theta[y]
    # print(x, y, tx, ty, W[x, y], theta[y], sA[ty], sB[ty])

    cuda.syncthreads()
    print(x, y, sA[0], sA[1], sB[0], sB[1])
    
        
    
    sum_sin_shared_mem = 0.0
    sum_cos_shared_mem = 0.0
    
    if ty == 0:
        for i in range(TPB):
            if (TPB * cuda.blockIdx.y + i) < theta.shape[0]:
                sum_sin_shared_mem += math.sin(sB[i]) * sA[i]
                sum_cos_shared_mem += math.cos(sB[i]) * sA[i]
        
        cuda.syncthreads()

        S[x, cuda.blockIdx.y] = sum_sin_shared_mem
        S[x, cuda.blockIdx.y] = sum_cos_shared_mem
        
    

def computing6(thetaBefore, W):
    theta = np.copy(thetaBefore)

    threadsperblock = (1, TPB)
    blockspergrid_x = math.ceil(W.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(W.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    S = np.empty([blockspergrid_x, blockspergrid_y])
    C = np.empty([blockspergrid_x, blockspergrid_y])

    d_Theta = cuda.to_device(theta)
    d_W = cuda.to_device(W)
    d_S = cuda.to_device(S)
    d_C = cuda.to_device(C)

    kernel_computing6[threadsperblock, blockspergrid](d_Theta, d_W, d_S, d_C)

    # get data
    processed_S = d_S.copy_to_host()
    processed_C = d_C.copy_to_host()

    finalS = np.zeros([theta.shape[0]])
    finalC = np.zeros([theta.shape[0]])
    
    for i in range(theta.shape[0]):
        for k in range(processed_S.shape[1]):
            finalS[i] += processed_S[i, k]
            finalC[i] += processed_C[i, k]

    return finalC, finalS


W = np.arange(9).reshape(3,3)
PI = np.pi
theta = 2 * PI * np.random.rand(3)
print("---------- weight")
print(W)
print("---------- theta")
print(theta)


C, S = computing6(theta, W)
