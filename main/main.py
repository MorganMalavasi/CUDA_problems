# In this file it is shown the computation of the euclidean distance

import math
import cupy as cp
from numba import jit, cuda, float32
from sklearn.metrics.pairwise import euclidean_distances
from numpy.testing import assert_almost_equal
# ignore warnings for jit
import warnings
warnings.filterwarnings("ignore")

TPB = 2 # thread per block

def distance_base(A, B):
    dist = (A**2).sum(axis=1)[:,np.newaxis] + (B**2).sum(axis=1) - 2*np.dot(A,B.T)
    # dist = sqrtZeroNan(dist)
    return dist

def checkResultsTable(array1, array2):
    # check sizes
    if array1.shape[0] != array2.shape[0] or array1.shape[1] != array2.shape[1]:
        return False

    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            
            # print("-> {0} - {1} = {2}".format(  abs(array1[i,j]), abs(array2[i,j]), abs(array1[i,j]) - abs(array2[i,j])) )
            
            if abs(abs(array1[i,j]) - abs(array2[i,j])) > 0.001 :
            # if abs(array1[i, j]) - abs(array2[i, j]) != 0:
                print(abs(array1[i, j]) - abs(array2[i, j]))
                return False
    return True

def checkResults(array1, array2):
    if array1.shape[0] != array2.shape[0]:
        return False
    
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
    import skcuda.linalg as culinalg
    from cpu_dist import dist as dist_cpu
    from gpu_dist import dist as dist_gpu
    from pycuda.compiler import SourceModule as SM
    from pycuda.cumath import sqrt as cusqrt
    import pycuda.gpuarray as gpuarray
    from pycuda.elementwise import ElementwiseKernel
  
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
    sample9, l9 = create_dataset_base(samples = 10000, features = 1024, centers = 5, display = False, n_dataset = 9)
    sample10, l10 = create_dataset_base(samples = 15000, features = 2048, centers = 5, display = False, n_dataset = 10)
    sample11, l11 = create_dataset_base(samples = 15500, features = 4096, centers = 8, display = False, n_dataset = 11)
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
    listOfDataset.append((sample3, l3))
    listOfDataset.append((sample4, l4))
    # listOfDataset.append((sample5, l5))
    # listOfDataset.append((sample6, l6))
    listOfDataset.append((sample7, l7))
    # listOfDataset.append((sample8, l8))
    listOfDataset.append((sample9, l9))
    # listOfDataset.append((sample10, l10))
    # listOfDataset.append((sample11, l11))

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
    @timeit
    def cpu_function(data, theta, eps = 0.01, normalize = True):

        # part 1
        matrixOfWeights = euclidean_distances(data, data)
        
        if normalize:
            matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)

        # part 2
        sin_t = np.sin(theta)
        S = np.dot(matrixOfWeights, sin_t)

        cos_t = np.cos(theta)
        C = np.dot(matrixOfWeights, cos_t)


        return matrixOfWeights, C, S
    

    ##########################################################################################
    # alg18
    @timeit
    def gpu_function(data, theta, eps = 0.01, normalize = True):
        
        # part 1
        matrixOfWeights = dist_gpu(data, data, optimize_level=3, output="gpu")

        mod = SM("""
            __global__ void diagonal_zeros(float *mat, int size)
            {
                int idx = threadIdx.x + blockDim.x * blockIdx.x;
                if (idx >= size){
                    return;
                }
                
                mat[idx * size + idx] = 0.0;
            }
            """)

        func = mod.get_function("diagonal_zeros")
        block = (32, 1, 1)
        grid_x = (matrixOfWeights.shape[0] + (block[0] - 1)) // block[0]
        grid_y = 1
        grid = (grid_x, grid_y)
        size = np.int32(matrixOfWeights.shape[0])
        func(matrixOfWeights, size, block = block, grid = grid)

        matrixOfWeights = cusqrt(matrixOfWeights)

        val = culinalg.norm(matrixOfWeights)
        _weights_ = matrixOfWeights / val

        # slower way
        '''
        kernel_division = ElementwiseKernel(
            "float *x, float val, float *z",
            "z[i] = x[i] / val",
            "kernel_division")

        _weights_ = gpuarray.empty_like(matrixOfWeights)
        kernel_division(matrixOfWeights, val, _weights_)
        '''

        # part 2
        # TODO -> to_gpu_async
        theta_sin = np.asarray(np.sin(theta) , np.float32)
        theta_cos = np.asarray(np.cos(theta) , np.float32)

        theta_sin_gpu = gpuarray.to_gpu(theta_sin)
        theta_cos_gpu = gpuarray.to_gpu(theta_cos)

        S_GPU = culinalg.dot(_weights_, theta_sin_gpu)
        C_GPU = culinalg.dot(_weights_, theta_cos_gpu)    

        return (_weights_.get(), C_GPU.get(), S_GPU.get())



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

        dataCPU = np.copy(eachTupleBeforeTransformation[0])
        thetaCPU = np.copy(eachTupleBeforeTransformation[2])
        dataGPU = np.copy(eachTupleBeforeTransformation[0])
        thetaGPU = np.copy(eachTupleBeforeTransformation[2])
        
        print("CPU ---------------------------------------------------------")
        matrix_cpu, C_cpu, S_cpu = cpu_function(dataCPU, thetaCPU)
        
        print("GPU ---------------------------------------------------------")
        matrix_gpu, C_gpu, S_gpu = gpu_function(dataGPU, thetaGPU)

        # print("gpu opt lev 4 ----------------------------------------------")
        # matrix19 = euclidean_distance_function19(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)

        
        # check results 
        # with np.printoptions(precision=8, suppress=True):        
        '''
        print("-------data--------")
        print(eachTupleBeforeTransformation[0])
        print("-------base--------")
        print(matrix_cpu)
        print("///////////////")
        print(matrix_gpu)
        '''
        

        # print(checkResultsTable(matrix_cpu, matrix_gpu))
        print(checkResults(C_cpu, C_gpu))
        print(checkResults(S_cpu, S_gpu))
        assert_almost_equal(matrix_cpu, matrix_gpu)
        assert_almost_equal(C_cpu, C_gpu)
        assert_almost_equal(S_cpu, S_gpu)
        print("TEST PASSED")
        