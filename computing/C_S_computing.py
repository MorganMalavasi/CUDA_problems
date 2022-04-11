# In this file it is shown the computation of the euclidean distance

import math
import cupy as cp
import numpy as np
from numba import jit, cuda, float32, float64
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Process, cpu_count, Pool, RawArray, Array
import threading
# ignore warnings for jit
import warnings
warnings.filterwarnings("ignore")

TPB = 32 # threads per block

def angle_computation(theta, W, function, arr):
    if function == "sin":
        val = np.sin(theta)
    elif function == "cos":
        val = np.cos(theta)
    arr[0:theta.shape[0]] = np.dot(W, val)
    
def angle_computation_divided_sin(theta, W, size, start, stop, M):
    val = np.sin(theta)
    M[start:stop] = np.dot(W[start:stop, 0:size], val)

def angle_computation_divided_cos(theta, W, size, start, stop, M):
    val = np.cos(theta)
    M[start:stop] = np.dot(W[start:stop, 0:size], val)

def angle_computation_divided_sin_cos(theta, W, size, start, stop, S, C):
    val = np.sin(theta)
    S[start:stop] = np.dot(W[start:stop, 0:size], val)
    val = np.cos(theta)
    C[start:stop] = np.dot(W[start:stop, 0:size], val)


def checkResults(array1, array2):
    '''
    if array1.shape[0] != array2.shape[0]:
        return False
    '''
    for i in range(array1.shape[0]):
        if abs(abs(array1[i]) - abs(array2[i])) > 0.0000001 :
            # if abs(array1[i, j]) - abs(array2[i, j]) != 0:
                print(abs(array1[i]) - abs(array2[i]))
                return False
    return True



def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%2.2f sec' % \
            (te-ts))
        return result

    return timed

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

if __name__ == '__main__': 
    import os, time, sys, time
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, load_iris, fetch_olivetti_faces
    from sklearn.decomposition import PCA
    from numpy import linalg
    
    from pycuda import cumath
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    culinalg.init()
    
    n_cores = cpu_count()
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
    sample6, l6 = create_dataset_base(samples = 15000, features = 30, centers = 7, display = False, n_dataset = 6)
    sample7, l7 = create_dataset_base(samples = 8000, features = 300, centers = 7, display = False, n_dataset = 7)   # -> !!!not working in cuda base 
    sample8, l8 = create_dataset_base(samples = 6, features = 5, centers = 2, display = False, n_dataset = 8)
    sample9, l9 = create_dataset_base(samples = 10000, features = 2048, centers = 5, display = False, n_dataset = 9)
    sample10, l10 = create_dataset_base(samples = 20000, features = 10, centers = 5, display = False, n_dataset = 10)
    # sample9, l9 = create_dataset_base(samples = 20000, features = 20, centers = 5, display = False, n_dataset = 9)
    # sample4, l4 = create_dataset_moon(samples = 1000, noise = 0.05, display = True, n_dataset = 4)
    # sample5, l5 = create_dataset_moon(samples = 1000, noise = 0.2, display = True, n_dataset = 5)
    # sample6, l6 = create_dataset_circles(samples = 1000, noise = 0.05, display = True, n_dataset = 6)
    # sample7, l7 = create_dataset_classification(n_samples = 100, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = True, n_dataset = 7)
    # sample8, l8 = create_dataset_classification(n_samples = 1000, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = True, n_dataset = 8)
    # sample9, l9 = create_dataset_olivetti_faces(display = False, n_dataset = 9)

    listOfDataset = []
    # listOfDataset.append((sample0, l0))
    # listOfDataset.append((sample1, l1))
    # listOfDataset.append((sample2, l2))
    # listOfDataset.append((sample3, l3))
    # listOfDataset.append((sample4, l4))
    # listOfDataset.append((sample5, l5))
    listOfDataset.append((sample6, l6))
    # listOfDataset.append((sample7, l7))
    # listOfDataset.append((sample8, l8))
    # listOfDataset.append((sample9, l9))
    # listOfDataset.append((sample10, l10))

    # creation of the starting points -> theta  |  for each dataset
    PI = np.pi
    ''' listOfInitialTuple is a list of tuples '''
    listOfInitialTuple = []
    for eachDataset in listOfDataset:
        numberOfSamplesInTheDataset = eachDataset[0].shape[0]
        # new array of theta for the points inside each dataset
        theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
        listOfInitialTuple.append((eachDataset[0], eachDataset[1], theta))

    # computing the weights
    def _euclidean_distance_(thetaToTransform, eps = 0.01, normalize = True):
        data = np.copy(thetaToTransform[0])
        matrixOfWeights = euclidean_distances(data, data)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)
        
        return matrixOfWeights

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ################### methods ##############################################################
    ##########################################################################################

    ##########################################################################################
    # alg1
    @timeit
    def computing1(thetaBefore, W):
        theta = np.copy(thetaBefore)

        sin_t = np.sin(theta)
        S = np.dot(W, sin_t)

        cos_t = np.cos(theta)
        C = np.dot(W, cos_t)
        
        return C,S

    ##########################################################################################
    # alg2
    @timeit
    def computing2(thetaBefore, W):
        theta = np.copy(thetaBefore)
        
        S = np.empty([theta.shape[0]])
        C = np.empty([theta.shape[0]])
        t1 = threading.Thread(target = angle_computation, args = (theta, W, "sin", S,  ))
        t1.start()
        t2 = threading.Thread(target = angle_computation, args = (theta, W, "cos", C,  ))
        t2.start()

        t1.join()
        t2.join()

        return C,S

    ##########################################################################################
    # alg3
    @timeit
    def computing3(thetaBefore, W):
        theta = np.copy(thetaBefore)
        
        S = np.empty([theta.shape[0]])
        C = np.empty([theta.shape[0]])
        
        size = theta.shape[0]

        _size_ = size // n_cores
        listOfThreads = []
        for i in range(n_cores):
            if i < n_cores-1:
                t = threading.Thread(target = angle_computation_divided_sin, args = (theta, W, size, i*_size_, (i+1)*_size_, S ))
            else:
                t = threading.Thread(target = angle_computation_divided_sin, args = (theta, W, size, i*_size_, size, S ))
            t.start()
            listOfThreads.append(t)

        for i in range(n_cores):
            if i < n_cores-1:
                t = threading.Thread(target = angle_computation_divided_cos, args = (theta, W, size, i*_size_, (i+1)*_size_, C ))
            else:
                t = threading.Thread(target = angle_computation_divided_cos, args = (theta, W, size, i*_size_, size, C ))
            t.start()                
            listOfThreads.append(t)

        for t in listOfThreads:
            t.join()
        
        return C,S

    ##########################################################################################
    # alg4
    @timeit
    def computing4(thetaBefore, W):
        theta = np.copy(thetaBefore)

        S = Array('d', theta.shape[0])
        C = Array('d', theta.shape[0])

        size = theta.shape[0]

        _size_ = size // n_cores
        listOfThreads = []
        
        for i in range(n_cores):
            if i < n_cores-1:
                t = Process(target = angle_computation_divided_sin_cos, args = (theta, W, size, i*_size_, (i+1)*_size_, S, C ))
            else:
                t = Process(target = angle_computation_divided_sin_cos, args = (theta, W, size, i*_size_, size, S, C ))
            t.start()
            listOfThreads.append(t)
        
        for t in listOfThreads:
            t.join()

        return C,S

    

    ##########################################################################################
    # alg5
    @cuda.jit
    def kernel_computing5(theta, W, S, C):
        pos = cuda.grid(1)

        if pos >= theta.shape[0]:
            return
        
        sumS = 0.0
        sumC = 0.0
        for i in range(theta.shape[0]):
            sumS += W[pos, i] * math.sin(theta[i])
            sumC += W[pos, i] * math.cos(theta[i])
        
        S[pos] = sumS
        C[pos] = sumC


    @timeit
    def computing5(thetaBefore, W):
        theta = np.copy(thetaBefore)

        S = np.empty([theta.shape[0]])
        C = np.empty([theta.shape[0]])
        
        size = theta.shape[0]
        
        d_Theta = cuda.to_device(theta)
        d_W = cuda.to_device(W)
        d_S = cuda.to_device(S)
        d_C = cuda.to_device(C)

        # call kernel
        threadsperblock = 32
        blockspergrid = (size + (threadsperblock-1)) // threadsperblock
        kernel_computing5[blockspergrid, threadsperblock](d_Theta, d_W, d_S, d_C)

        # get data
        processed_S = d_S.copy_to_host()
        processed_C = d_C.copy_to_host()
        
        return processed_C, processed_S

    ##########################################################################################
    # alg6
    @cuda.jit
    def kernel_computing6(theta, W, S, C):
        sA = cuda.shared.array(shape = (TPB), dtype = float32)
        sB = cuda.shared.array(shape = (TPB), dtype = float32)

        x = cuda.grid(1)
        # print(x)

        tx = cuda.threadIdx.x

        
        bpg = (W.shape[1] + (TPB-1)) // TPB

        sum_sin = 0.0
        sum_cos = 0.0
        for i in range(bpg):
            sA[tx] = W[x // TPB, i*TPB + tx]
            sB[tx] = theta[i*TPB + tx]

            cuda.syncthreads()

            # print(x, sA[tx], sB[tx])

            for j in range(TPB):
                if i*TPB + j < W.shape[1]:
                    sum_sin += math.sin(sB[j]) * sA[j]
                    sum_cos += math.cos(sB[j]) * sA[j]

            cuda.syncthreads()
        
        C[x//TPB] = sum_cos
        S[x//TPB] = sum_sin
        
        
    
    @timeit
    def computing6(thetaBefore, W):
        theta = np.copy(thetaBefore)

        S = np.empty([theta.shape[0]])
        C = np.empty([theta.shape[0]])

        d_Theta = cuda.to_device(theta)
        d_W = cuda.to_device(W)
        d_S = cuda.to_device(S)
        d_C = cuda.to_device(C)

        threadsperblock = TPB
        blockspergrid = theta.shape[0]
        kernel_computing6[blockspergrid, threadsperblock](d_Theta, d_W, d_S, d_C)

        # get data
        processed_S = d_S.copy_to_host()
        processed_C = d_C.copy_to_host()

        return processed_C, processed_S

    ##########################################################################################
    # alg7

    threadsperblock_dot_prod = 64

    @cuda.jit
    def kernel_dot_prod(theta, line, S, C, absolute_position):
        pos = cuda.grid(1)

        if pos >= theta.shape[0]:
            return

        sMEM_sin = cuda.shared.array(shape = (threadsperblock_dot_prod), dtype = float32)
        sMEM_cos = cuda.shared.array(shape = (threadsperblock_dot_prod), dtype = float32)
        sMEM_sin[cuda.threadIdx.x] = math.sin(theta[pos]) * line[pos]
        sMEM_cos[cuda.threadIdx.x] = math.cos(theta[pos]) * line[pos]

        cuda.syncthreads()

        if cuda.threadIdx.x == 0:
            sum_sin = 0.0
            sum_cos = 0.0
            for i in range(threadsperblock_dot_prod):
                if cuda.blockDim.x * cuda.blockIdx.x + i < theta.shape[0]:
                    sum_sin += sMEM_sin[i]
                    sum_cos += sMEM_cos[i]

            cuda.syncthreads()
            
            cuda.atomic.add(S, absolute_position, sum_sin)
            cuda.atomic.add(C, absolute_position, sum_cos)
            

    @timeit
    def computing7(thetaBefore, W):
        theta = np.copy(thetaBefore)

        S = np.zeros([theta.shape[0]])
        C = np.zeros([theta.shape[0]])

        blockspergrid_dot_prod = (W.shape[1] + (threadsperblock_dot_prod - 1)) // threadsperblock_dot_prod

        # numba doesn't implement dynamic parallelism
        stream = cuda.stream()
        d_Theta = cuda.to_device(theta, stream = stream)
        d_S = cuda.to_device(S, stream = stream)
        d_C = cuda.to_device(C, stream = stream)
        
        for i in range(W.shape[0]):
            line = W[i,:]
            d_line = cuda.to_device(line, stream = stream)
            kernel_dot_prod[blockspergrid_dot_prod, threadsperblock_dot_prod, stream](d_Theta, d_line, d_S, d_C, i)
            # get data       

        stream.synchronize() 
        d_S.copy_to_host(S, stream = stream)
        d_C.copy_to_host(C, stream = stream) 

        return C, S


    ##########################################################################################
    # alg7bis

    threadsperblock_tuple = (1, 32)

    @cuda.jit
    def kernel_computing7bis(theta, W, S, C):
        row, col = cuda.grid(2)

        if col >= W.shape[1]:
            return
        
        _TPB_ = threadsperblock_tuple[1]
        
        sMEM_sin = cuda.shared.array(shape = (_TPB_), dtype = float32)
        sMEM_cos = cuda.shared.array(shape = (_TPB_), dtype = float32)
        sMEM_sin[cuda.threadIdx.y] = math.sin(theta[col]) * W[row, col]
        sMEM_cos[cuda.threadIdx.y] = math.cos(theta[col]) * W[row, col]

        cuda.syncthreads()

        if cuda.threadIdx.y == 0:
            sum_sin = 0.0
            sum_cos = 0.0
            for i in range(_TPB_):
                if cuda.blockDim.y * cuda.blockIdx.y + i < W.shape[1]:
                    sum_sin += sMEM_sin[i]
                    sum_cos += sMEM_cos[i]
            cuda.syncthreads()
            
            cuda.atomic.add(S, row, sum_sin)
            cuda.atomic.add(C, row, sum_cos)

    @timeit
    def computing7bis(thetaBefore, W):
        theta = np.copy(thetaBefore)
    
        S = np.zeros([theta.shape[0]])
        C = np.zeros([theta.shape[0]])

        d_Theta = cuda.to_device(theta)
        d_W = cuda.to_device(W)
        d_S = cuda.to_device(S)
        d_C = cuda.to_device(C)

        blockspergrid_x = W.shape[0]
        blockspergrid_y = (W.shape[1] + (threadsperblock_tuple[1] - 1)) // threadsperblock_tuple[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        start = time.time()
        kernel_computing7bis[blockspergrid, threadsperblock_tuple](d_Theta, d_W, d_S, d_C)
        end = time.time()
        print(' tempo interno = %2.2f sec' % (end-start))
        # get data
        processed_S = d_S.copy_to_host()
        processed_C = d_C.copy_to_host()

        return processed_C, processed_S
    

    ##########################################################################################
    # alg8
    @timeit
    def computing8(thetaBefore, W):
        theta = np.copy(thetaBefore)

        theta_sin = np.sin(theta)
        theta_cos = np.cos(theta)

        theta_sin_gpu = gpuarray.to_gpu(theta_sin)
        theta_cos_gpu = gpuarray.to_gpu(theta_cos)
        weight_gpu = gpuarray.to_gpu(W)

        S_GPU = culinalg.dot(weight_gpu, theta_sin_gpu)
        C_GPU = culinalg.dot(weight_gpu, theta_cos_gpu)

        C_GPU = C_GPU.get()
        S_GPU = S_GPU.get()

        return C_GPU, S_GPU


    ##########################################################################################
    # alg9
    @timeit
    def computing9(thetaBefore, W):
        theta = np.copy(thetaBefore)

        theta_gpu = gpuarray.to_gpu(theta)
        weight_gpu = gpuarray.to_gpu(W)

        theta_sin_gpu = cumath.sin(theta_gpu)
        theta_cos_gpu = cumath.cos(theta_gpu)

        S_GPU = culinalg.dot(weight_gpu, theta_sin_gpu)
        C_GPU = culinalg.dot(weight_gpu, theta_cos_gpu)

        C_GPU = C_GPU.get()
        S_GPU = S_GPU.get()

        return C_GPU, S_GPU


        
    # process the data
    counter = 0
    for eachTupleBeforeTransformation in listOfInitialTuple:
        counter += 1

        weights = _euclidean_distance_(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001, normalize = True)

        print("DATABASE : {}".format(counter))
        print("base ----------------------------------------------------------")
        results1C, results1S = computing1(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("multithread ---------------------------------------------------")
        results2C, results2S = computing2(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("multithread strong --------------------------------------------")
        results3C, results3S = computing3(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("multithread shared mem ----------------------------------------")

        # kernels
        '''
        results4C, results4S = computing4(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("kernel numba simple -------------------------------------------")
        results5C, results5S = computing5(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("kernel numba shared mem ---------------------------------------")
        results6C, results6S = computing6(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("kernel dot prod shared mem ------------------------------------")
        results7C, results7S = computing7(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("kernel dot prod shared mem matrix -----------------------------")
        results7bisC, results7bisS = computing7bis(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        '''
        print("cublas --------------------------------------------------------")
        results8C, results8S = computing8(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        print("cublas optimized trigonometric functions ----------------------")
        results9C, results9S = computing9(thetaBefore = eachTupleBeforeTransformation[2], W = weights)
        
        # check results 
        
        print(checkResults(results1C, results9C))
        print(checkResults(results1S, results9S))
        