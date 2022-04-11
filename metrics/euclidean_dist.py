# In this file it is shown the computation of the euclidean distance

import math
import cupy as cp
from numba import jit, cuda, float32
from sklearn.metrics.pairwise import euclidean_distances
# ignore warnings for jit
import warnings
warnings.filterwarnings("ignore")

TPB = 2 # thread per block

def distance_base(A, B):
    dist = (A**2).sum(axis=1)[:,np.newaxis] + (B**2).sum(axis=1) - 2*np.dot(A,B.T)
    # dist = sqrtZeroNan(dist)
    return dist

def distance_einsum(A, B):
    dist = np.einsum('ij,ij->i',A,A)[:,np.newaxis] \
     + np.einsum('ij,ij->i',B,B) \
     - 2*np.dot(A,B.T)
    # dist = sqrtZeroNan(dist)
    return dist

@cuda.jit
def dot_product(arr1, arr2):
    tot = 0
    length = arr1.shape[0]
    for i in range(length):
      tot += arr1[i] * arr2[i]
    return tot

def loop_euclidean_distance(X, size, start, stop, matrixOfWeights):
    for k in range(start, stop):
        lineX = X[k,:]
        for j in range(size):
            lineY = X[j,:]
            _1 = dot_product(lineX, lineX)
            _2 = 2*dot_product(lineX, lineY)
            _3 = dot_product(lineY, lineY)
            matrixOfWeights[k, j] = math.sqrt(_1 - _2 + _3)

@jit
def dot_product_jit(arr1, arr2):
    tot = 0
    length = arr1.shape[0]
    for i in range(length):
        tot += arr1[i] * arr2[i]
    return tot

@jit
def loop_euclidean_distance_jit(X, size, start, stop, matrixOfWeights):
    for k in range(start, stop):
        lineX = X[k,:]
        for j in range(size):
            lineY = X[j,:]
            _1 = dot_product_jit(lineX, lineX)
            _2 = 2*dot_product_jit(lineX, lineY)
            _3 = dot_product_jit(lineY, lineY)
            matrixOfWeights[k, j] = math.sqrt(_1 - _2 + _3)

def pairwise_euclidean_distance_multithread(X, Y, col, start, stop, finalMat):
    uploadMatrix = X[start:stop, 0:col]
    M = euclidean_distances(uploadMatrix, Y)
    row = M.shape[0]
    col = M.shape[1]

    finalMat[start:stop, 0:finalMat.shape[1]] = M

    '''
    for i in range(row):
        for j in range(col):
            finalMat[i + start, j] = M[i, j]
    '''

def pairwise_euclidean_distance_gemm_multithread(X, Y, size, start, stop, finalMat):
    uploadMatrix = X[start:stop, 0:size]
    M = dist_cpu(uploadMatrix, Y, matmul='gemm', method='ext', precision='float32')
    row = M.shape[0]
    col = M.shape[1]

    finalMat[start:stop, 0:finalMat.shape[1]] = M

def checkResultsTable(array1, array2):
    # check sizes
    if array1.shape[0] != array2.shape[0] or array1.shape[1] != array2.shape[1]:
        return False

    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            
            # print("-> {0} - {1} = {2}".format(  abs(array1[i,j]), abs(array2[i,j]), abs(array1[i,j]) - abs(array2[i,j])) )
            
            if abs(abs(array1[i,j]) - abs(array2[i,j])) > 0.0000001 :
            # if abs(array1[i, j]) - abs(array2[i, j]) != 0:
                print(abs(array1[i, j]) - abs(array2[i, j]))
                return False
    return True

def sqrtZeroNan(ini_array):
    for i in range(ini_array.shape[0]):
        for j in range(ini_array.shape[1]):
            if ini_array[i,j] > -0.00001 and ini_array[i,j] < 0.00001:
                ini_array[i,j] = 0.0

    result = np.sqrt(ini_array)
    b = np.where(np.isnan(result), 0, result)
    return b

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%2.2f sec' % \
            (te-ts))
        return result

    return timed

if __name__ == '__main__': 
  
    import numpy as np
    import os, time, sys, time
    import threading, multiprocessing
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, load_iris, fetch_olivetti_faces
    from sklearn.decomposition import PCA
    from numpy import linalg
    from cpu_dist import dist as dist_cpu
    from gpu_dist import dist as dist_gpu
  
    n_cores = multiprocessing.cpu_count()
    print('cores CPU = {0}'.format(n_cores))
    print(cuda.detect())
    list_results = []
    
    ''' generation of the datasets'''

    def create_dataset_iris(display = False, n_dataset = 0, stardard = True):
        data = load_iris()
        X = data.data
        l = data.target
        if stardard:
            X = scaling(X)
        if display:
            doPCA(X, l, n_dataset)
        return X, l

    def create_dataset_base(samples, features, centers, standard_deviation_cluster = 1, standard = True, display = False, n_dataset = 0):
        # X = The generated samples
        # l = The integer labels for cluster membership of each sample
        X, l = make_blobs(n_samples = samples, n_features = features, centers = centers, cluster_std = standard_deviation_cluster, random_state = None)
        if standard:
            X = scaling(X)
        if display:
            doPCA(X, l, n_dataset)
        return X, l

    def create_dataset_moon(samples, noise, standard = True, display = False, n_dataset = 0):
        X, l = make_moons(n_samples = samples, noise = noise)
        if standard:
            X = scaling(X)
        if display:
            doPCA(X, l, n_dataset)
        return X, l 

    def create_dataset_circles(samples, noise, standard = True, display = False, n_dataset = 0):
        X, l = make_circles(n_samples = samples, noise = noise)
        if standard:
            X = scaling(X)
        if display:
            doPCA(X, l, n_dataset)
        return X, l 

    def create_dataset_classification(n_samples, n_features, n_redundant, n_informative, n_clustes_per_class, display = False, n_dataset = 0, standard = True):
        X, l = make_classification(n_samples = n_samples, n_features = n_features, n_redundant = n_redundant, n_informative = n_informative, n_clusters_per_class=n_clustes_per_class)
        if standard:
            X = scaling(X)
        if display:
            doPCA(X, l, n_dataset)
        return X, l

    def create_dataset_olivetti_faces(display = False, n_dataset = 0, standard = True):
        data = fetch_olivetti_faces()
        X = data.data
        l = data.target
        if standard:
            X = scaling(X)
        if display:
            doPCA(X, l, n_dataset)
        return X, l

    def scaling(samples):
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        return samples
      
    def doPCA(X, labels, n_dataset):
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        fig = px.scatter(components, x = 0, y = 1, title = 'Blobs', color=labels, labels={'0': 'PC 1', '1': 'PC 2'})
        fig.update_layout(
            width = 600,
            height = 600,
            title = 'Dataset nr {0} - samples = {1} - features = {2} - classes = {3}'.format(n_dataset, X.shape[0], X.shape[1], np.max(labels) + 1))
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1)
        fig.show()

    sample0, l0 = create_dataset_iris(display = False, n_dataset = 0)
    sample1, l1 = create_dataset_base(samples = 1000, features = 3, centers = 2, display = False, n_dataset = 1)
    sample2, l2 = create_dataset_base(samples = 5000, features = 7, centers = 10, display = True, n_dataset = 2)
    sample3, l3 = create_dataset_base(samples = 7000, features = 30, centers = 10, display = True, n_dataset = 3)
    sample4, l4 = create_dataset_base(samples = 7000, features = 50, centers = 8, display = False, n_dataset = 4)
    sample5, l5 = create_dataset_base(samples = 10000, features = 20, centers = 10, display = False, n_dataset = 5)
    sample6, l6 = create_dataset_base(samples = 15000, features = 5, centers = 7, display = False, n_dataset = 6)
    sample7, l7 = create_dataset_base(samples = 8000, features = 300, centers = 7, display = False, n_dataset = 7)   # -> !!!not working in cuda base 
    sample8, l8 = create_dataset_base(samples = 5, features = 4, centers = 2, display = False, n_dataset = 8)
    sample9, l9 = create_dataset_base(samples = 10000, features = 2048, centers = 5, display = False, n_dataset = 9)
    # sample9, l9 = create_dataset_base(samples = 20000, features = 20, centers = 5, display = False, n_dataset = 9)
    # sample4, l4 = create_dataset_moon(samples = 1000, noise = 0.05, display = True, n_dataset = 4)
    # sample5, l5 = create_dataset_moon(samples = 1000, noise = 0.2, display = True, n_dataset = 5)
    # sample6, l6 = create_dataset_circles(samples = 1000, noise = 0.05, display = True, n_dataset = 6)
    # sample7, l7 = create_dataset_classification(n_samples = 100, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = True, n_dataset = 7)
    # sample8, l8 = create_dataset_classification(n_samples = 1000, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = True, n_dataset = 8)
    # sample9, l9 = create_dataset_olivetti_faces(display = False, n_dataset = 9)

    listOfDataset = []
    # listOfDataset.append((sample0, l0))
    # listOfDataset.append((sample1, l1))
    # listOfDataset.append((sample2, l2))
    # listOfDataset.append((sample3, l3))
    # listOfDataset.append((sample4, l4))
    # listOfDataset.append((sample5, l5))
    # listOfDataset.append((sample6, l6))
    listOfDataset.append((sample7, l7))
    # listOfDataset.append((sample8, l8))
    # listOfDataset.append((sample9, l9))

    # creation of the starting points -> theta  |  for each dataset
    PI = np.pi
    ''' listOfInitialTuple is a list of tuples '''
    listOfInitialTuple = []
    for eachDataset in listOfDataset:
        numberOfSamplesInTheDataset = eachDataset[0].shape[0]
        # new array of theta for the points inside each dataset
        theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
        listOfInitialTuple.append((eachDataset[0], eachDataset[1], theta))


    ##########################################################################################
    # alg1 
    @timeit
    def euclidean_distance_function1(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = euclidean_distances(data, data)
        
        '''
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)
        '''
        return matrixOfWeights

    ##########################################################################################
    # alg2
    @timeit
    def euclidean_distance_function2(thetaToTransform, eps = 0.01, normalize = True):
        X = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        col = thetaToTransform[0].shape[1]
        matrixOfWeights = np.empty([size, size])

        _size_ = size // n_cores
        listOfThreads = []
        for i in range(n_cores):
            if i < n_cores-1:
                t = multiprocessing.Process(target = pairwise_euclidean_distance_multithread, args = (X, X, col, i*_size_, (i+1)*_size_, matrixOfWeights, ))
            else:
                t = multiprocessing.Process(target = pairwise_euclidean_distance_multithread, args = (X, X, col, i*_size_, size, matrixOfWeights, ))
            listOfThreads.append(t)

        for t in listOfThreads:
            t.start()
    
        for t in listOfThreads:
            t.join()
      
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    

        return matrixOfWeights

    ##########################################################################################
    # alg3
    @timeit
    @jit
    def euclidean_distance_function3(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        # matrixOfWeights = euclidean_distances(data, data)
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])
        loop_euclidean_distance_jit(data, data.shape[0], 0, data.shape[0], matrixOfWeights)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)
        
        return matrixOfWeights

    ##########################################################################################
    # alg4 -> kernel simple
    @cuda.jit
    def kernel_euclidean_distance_function4(A, matrixOfWeights):
        pos = cuda.grid(1)

        '''
        for k in range(start, stop):
            lineX = X[k,:]
                for j in range(size):
                    lineY = X[j,:]
                    _1 = dot_product(lineX, lineX)
                    _2 = 2*dot_product(lineX, lineY)
                    _3 = dot_product(lineY, lineY)
                    matrixOfWeights[k, j] = math.sqrt(_1 - _2 + _3)
        '''
        if pos < A.shape[0]:
            lineX = A[pos, :]
            for k in range(A.shape[0]):
                lineY = A[k, :]
                _1 = dot_product(lineX, lineX)
                _2 = 2*dot_product(lineX, lineY)
                _3 = dot_product(lineY, lineY)
                matrixOfWeights[pos, k] = math.sqrt(_1 - _2 + _3)


    @timeit
    def euclidean_distance_function4(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])
        
        # load data into device memory
        d_A = cuda.to_device(data)
        d_C = cuda.to_device(matrixOfWeights)

        '''
            threadsperblock = 32
            blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
            increment_by_one[blockspergrid, threadsperblock](an_array)
        '''
        # call kernel
        threadsperblock = 32
        blockspergrid = (data.shape[0] + (threadsperblock - 1)) // threadsperblock
        kernel_euclidean_distance_function4[blockspergrid, threadsperblock](d_A, d_C)

        # get data from device memory
        processedMatrixOfWeights = d_C.copy_to_host()

        if normalize:
            processedMatrixOfWeights = processedMatrixOfWeights / linalg.norm(processedMatrixOfWeights)    

        return processedMatrixOfWeights

    ##########################################################################################
    # alg5 -> kernel using matrices
    @cuda.jit
    def kernel_euclidean_distance_function5(A, matrixOfWeights):
        row, col = cuda.grid(2)

        if row < matrixOfWeights.shape[0] and col < matrixOfWeights.shape[1]:
            lineX = A[row, :]
            lineY = A[col, :]
            _1 = dot_product(lineX, lineX)
            _2 = 2*dot_product(lineX, lineY)
            _3 = dot_product(lineY, lineY)
            matrixOfWeights[row, col] = math.sqrt(_1 - _2 + _3)


    @timeit
    def euclidean_distance_function5(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])

        # load data into device memory
        d_A = cuda.to_device(data)
        d_C = cuda.to_device(matrixOfWeights)

        '''
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            increment_a_2D_array[blockspergrid, threadsperblock](an_array)
        '''

        threadsperblock = (16,16)
        blockspergrid_x = math.ceil(matrixOfWeights.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(matrixOfWeights.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        kernel_euclidean_distance_function5[blockspergrid, threadsperblock](d_A, d_C)

        # get data from device memory
        processedMatrixOfWeights = d_C.copy_to_host()

        if normalize:
            processedMatrixOfWeights = processedMatrixOfWeights / linalg.norm(processedMatrixOfWeights)

        return processedMatrixOfWeights

    ##########################################################################################
    # alg6 -> kernel simple with streams on memory

    @timeit
    def euclidean_distance_function6(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])

        # devide computation in 4 streams 
        stream = cuda.stream()
        with stream.auto_synchronize():
            d_A = cuda.to_device(data, stream = stream)
            d_C = cuda.to_device(matrixOfWeights, stream = stream)

            threadsperblock = 32
            blockspergrid = (data.shape[0] + (threadsperblock - 1)) // threadsperblock
            kernel_euclidean_distance_function4[blockspergrid, threadsperblock, stream](d_A, d_C)

            # get data from device memory
            d_C.copy_to_host(matrixOfWeights, stream = stream)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights


    ##########################################################################################
    # alg7 -> kernel using matrices and streams
    @timeit
    def euclidean_distance_function7(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])

        # devide computation in 4 streams 
        stream = cuda.stream()
        with stream.auto_synchronize():
            d_A = cuda.to_device(data, stream = stream)
            d_C = cuda.to_device(matrixOfWeights, stream = stream)

            threadsperblock = (16,16)
            blockspergrid_x = math.ceil(matrixOfWeights.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(matrixOfWeights.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            kernel_euclidean_distance_function5[blockspergrid, threadsperblock, stream](d_A, d_C)

            # get data from device memory
            d_C.copy_to_host(matrixOfWeights, stream = stream)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    ##########################################################################################
    # alg8 -> kernel using matrices streams and dot product in device memory
    
    @cuda.jit
    def kernel_dot_product(lineX, lineY, result, pos):
        tot = 0
        for i in range(len(lineX)):
            tot += lineX[i] * lineY[i]
        result[pos] = tot

    @cuda.jit
    def kernel_euclidean_distance_function8(A, matrixOfWeights):
        row, col = cuda.grid(2)

        if row < matrixOfWeights.shape[0] and col < matrixOfWeights.shape[1]:
            
            lineX = A[row, :]
            lineY = A[col, :]
            result = np.empty([3])

            stream_dot = cuda.stream()
            
            d_lineX = cuda.to_device(lineX)
            d_lineY = cuda.to_device(lineY)
            d_result = cuda.to_device(result)
            
            kernel_dot_product[1, 1, stream_dot](d_lineX, d_lineX, d_result, 0)
            kernel_dot_product[1, 1, stream_dot](d_lineX, d_lineY, d_result, 1)
            kernel_dot_product[1, 1, stream_dot](d_lineY, d_lineY, d_result, 2)

            stream_dot.synchronize()

            d_result.copy_to_host(result, stream = stream)

            _1 = result[0]
            _2 = result[1]
            _3 = result[2]
            # _1 - _2 + _3
            matrixOfWeights[row, col] = math.sqrt(_1 - _2 + _3)
    
    @timeit
    def euclidean_distance_function8(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])

        # devide computation in 4 streams 
        stream = cuda.stream()
        with stream.auto_synchronize():
            d_A = cuda.to_device(data, stream = stream)
            d_C = cuda.to_device(matrixOfWeights, stream = stream)

            threadsperblock = (16,16)
            blockspergrid_x = math.ceil(matrixOfWeights.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(matrixOfWeights.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            kernel_euclidean_distance_function5[blockspergrid, threadsperblock, stream](d_A, d_C)

            # get data from device memory
            d_C.copy_to_host(matrixOfWeights, stream = stream)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights


    ##########################################################################################
    # alg9 -> kernel simple with streams on memory and computation      -> DA AGGIUSTARE 
    @cuda.jit
    def kernel_euclidean_distance_function9(A, B, matrixOfWeights):
        pos = cuda.grid(1)
        #position = pos[1] + (i * A.shape[0])
        length = A.shape[0]

        if pos < length:
            lineX = A[pos, :]
            for k in range(B.shape[0]):
                lineY = B[k, :]
                _1 = dot_product(lineX, lineX)
                _2 = 2*dot_product(lineX, lineY)
                _3 = dot_product(lineY, lineY)
                matrixOfWeights[pos, k] = math.sqrt(_1 - _2 + _3)
    
    @timeit
    def euclidean_distance_function9(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0], data.shape[0]])

        # devide computation in 4 streams 
        cuda_streams = 4
        _size_ = data.shape[0] // cuda_streams

        stream = cuda.stream()
        d_B = cuda.to_device(data, stream = stream)
        d_C = cuda.to_device(matrixOfWeights, stream = stream)
        for i in range(cuda_streams):
            if i < cuda_streams - 1:
                subData = data[i * _size_ : (i+1) * _size_ , :]
                d_A = cuda.to_device(subData, stream = stream)
                threadsperblock = 32
                blockspergrid = (subData.shape[0] + (threadsperblock - 1)) // threadsperblock
                kernel_euclidean_distance_function9[blockspergrid, threadsperblock, stream](d_A, d_B, d_C)

            else:
                subData = data[i * _size_ : data.shape[0] , :]
                d_A = cuda.to_device(subData, stream = stream)
                threadsperblock = 32
                blockspergrid = (subData.shape[0] + (threadsperblock - 1)) // threadsperblock
                kernel_euclidean_distance_function9[blockspergrid, threadsperblock, stream](d_A, d_B, d_C)

        stream.synchronize()        
        
        # get data from device memory
        d_C.copy_to_host(matrixOfWeights, stream = stream)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    
    ##########################################################################################
    '''
    # alg9 -> kernel with shared memory
    @cuda.jit
    def kernel_shared_memory_euclidean_distance_function10(A, B, C):
        sA = cuda.shared.array(shape = (TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape = (TPB, TPB), dtype=float32)

        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = A.shape[0] // cuda.blockDim.x   # blocks per grid
        
        _1 = 0.0
        _2 = 0.0
        _3 = 0.0
        for i in range(bpg):
            sA[tx, ty] = A[x, ty + i * TPB]
            sB[ty, tx] = B[y, tx + i * TPB]

            cuda.syncthreads()
            if (x == 3) and (y == 0) :
                print("")
                print("matrix = sA")
                print(sA[0,0] , "|", sA[0,1])
                print("--------------------")
                print(sA[1,0] , "|", sA[1,1])
                print("")
                print("matrix = sB")
                print(sB[0,0] , "|", sB[0,1])
                print("--------------------")
                print(sB[1,0] , "|", sB[1,1])
                print("")


            for k in range(TPB):            
                _1 += sA[tx, k] * sA[tx, k]
                _2 += sA[tx, k] * sB[tx, k]
                _3 += sB[tx, k] * sB[tx, k]
                if (x == 3) and (y == 0) :
                    print("././././././././././././././")
                    print("sA[tx, k] = " , sA[tx, k])
                    print("_1 = " , _1)
            
            cuda.syncthreads()
        
        if x < C.shape[0] and y < C.shape[1]:
            C[x, y] = math.sqrt(_1 - 2*_2 + _3)


    @timeit
    def euclidean_distance_function10(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = np.empty([data.shape[0] , data.shape[0]])

        print(data)
    
        d_A = cuda.to_device(data)
        d_B = cuda.to_device(data)
        d_C = cuda.to_device(matrixOfWeights)

        # 16 x 16 threads per block
        threadsperblock = (TPB,TPB)
        blockspergrid_x = math.ceil(matrixOfWeights.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(matrixOfWeights.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        kernel_shared_memory_euclidean_distance_function10[blockspergrid, threadsperblock](d_A, d_B, d_C)

        # get data from device memory
        processedMatrixOfWeights = d_C.copy_to_host()
    
        if normalize:
            processedMatrixOfWeights = processedMatrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights
    
    '''    
    
    ##########################################################################################
    # alg11
    @timeit
    def euclidean_distance_function11(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = distance_base(data, data)

        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        '''        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)
        '''
        return matrixOfWeights

    ##########################################################################################
    # alg12
    @timeit
    def euclidean_distance_function12(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = distance_einsum(data, data)
        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)
        
        return matrixOfWeights

    ##########################################################################################
    # alg13
    @timeit
    def euclidean_distance_function13(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = dist_cpu(data, data, matmul='gemm', method='ext', precision='float32')
        # matrixOfWeights = sqrtZeroNan(matrixOfWeights)
        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    ''' -> not correct 
    ##########################################################################################
    # alg14
    @timeit
    def euclidean_distance_function14(thetaToTransform, eps = 0.01, normalize = True):
        X = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        matrixOfWeights = np.empty([size, size])

        _size_ = size // n_cores
        listOfThreads = []
        for i in range(n_cores):
            if i < n_cores-1:
                t = multiprocessing.Process(target = pairwise_euclidean_distance_gemm_multithread, args = (X, X, size, i*_size_, (i+1)*_size_, matrixOfWeights, ))
            else:
                t = multiprocessing.Process(target = pairwise_euclidean_distance_gemm_multithread, args = (X, X, size, i*_size_, (i+1)*_size_, matrixOfWeights, ))
        listOfThreads.append(t)

        for t in listOfThreads:
            t.start()
    
        for t in listOfThreads:
            t.join()
      
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    

        return matrixOfWeights
    
    '''

    ##########################################################################################
    # alg15
    @timeit
    def euclidean_distance_function15(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        
        matrixOfWeights = dist_gpu(data, data, optimize_level=0, output="cpu")

        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    ##########################################################################################
    # alg16
    @timeit
    def euclidean_distance_function16(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        
        matrixOfWeights = dist_gpu(data, data, optimize_level=1, output="cpu")

        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    ##########################################################################################
    # alg17
    @timeit
    def euclidean_distance_function17(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        
        matrixOfWeights = dist_gpu(data, data, optimize_level=2, output="cpu")

        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    ##########################################################################################
    # alg18
    @timeit
    def euclidean_distance_function18(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        
        matrixOfWeights = dist_gpu(data, data, optimize_level=3, output="cpu")

        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        return matrixOfWeights

    ##########################################################################################
    # alg19
    @timeit
    def euclidean_distance_function19(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        size = thetaToTransform[0].shape[0]
        
        matrixOfWeights = dist_gpu(data, data, optimize_level=4, output="cpu")

        for i in range(matrixOfWeights.shape[0]):
            matrixOfWeights[i, i] = 0.0

        '''
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)
        '''
        return matrixOfWeights



    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # process the data 
    counter = 0
    for eachTupleBeforeTransformation in listOfInitialTuple:
        counter += 1
        print("DATABASE : {}".format(counter))
        print("base ---------------------------------------------------------")
        matrix1 = euclidean_distance_function1(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("multicore ----------------------------------------------------")
        matrix2 = euclidean_distance_function2(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        '''
        print("jit not compiled ---------------------------------------------")
        matrix3 = euclidean_distance_function3(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("jit compiled -------------------------------------------------")
        matrix3 = euclidean_distance_function3(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda base ----------------------------------------------------")
        matrix4 = euclidean_distance_function4(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda matrix --------------------------------------------------")
        matrix5 = euclidean_distance_function5(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda base + stream on mem ------------------------------------")
        matrix6 = euclidean_distance_function6(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda matrix + stream on mem-----------------------------------")
        matrix7 = euclidean_distance_function7(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda matrix + stream on mem + dot_product --------------------")
        matrix8 = euclidean_distance_function8(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda matrix + stream on mem and compute + dot_product --------")
        matrix9 = euclidean_distance_function9(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("cuda kernel with shared memory -------------------------------")
        matrix10 = euclidean_distance_function10(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        '''
        print("accumulation_based_numpy I -----------------------------------")
        matrix11 = euclidean_distance_function11(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001) 
        print("accumulation_based_numpy II-----------------------------------")
        matrix12 = euclidean_distance_function12(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)   
        print("gemm_based_numpy ---------------------------------------------")
        matrix13 = euclidean_distance_function13(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)  
        '''
        print("gemm_based_numpy multicore------------------------------------")
        matrix14 = euclidean_distance_function14(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)  
        '''

        print("gpu opt lev 0 ------------------------------------------------")
        matrix15 = euclidean_distance_function15(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("gpu opt lev 1 ------------------------------------------------")
        matrix16 = euclidean_distance_function16(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("gpu opt lev 2 ------------------------------------------------")
        matrix17 = euclidean_distance_function17(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("gpu opt lev 3 ------------------------------------------------")
        matrix18 = euclidean_distance_function18(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
        print("gpu opt lev 4 ------------------------------------------------")
        matrix19 = euclidean_distance_function19(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)

        
        
        # check results 
        # with np.printoptions(precision=8, suppress=True):
        
        print("-------data--------")
        print(eachTupleBeforeTransformation[0])
        print("---------------")
        print(matrix1)
        print("///////////////")
        print(matrix11)
        print("///////////////")
        print(matrix19)
        
        print(checkResultsTable(matrix11, matrix19))  
        