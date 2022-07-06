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
from numpy.testing import assert_almost_equal


def checkResults(array1, array2):
    '''
    if array1.shape[0] != array2.shape[0]:
        return False
    '''
    for i in range(array1.shape[0]):
        if abs(abs(array1[i]) - abs(array2[i])) > 0.0001 :
            # if abs(array1[i, j]) - abs(array2[i, j]) != 0:
                print(abs(array1[i]) - abs(array2[i]))
                print("index = ", i)
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
    import skcuda.linalg as culinalg
    import skcuda.misc as misc
    from gpu_dist import dist as dist_gpu
    from pycuda.compiler import SourceModule as SM
    from pycuda.compiler import DynamicSourceModule
    from pycuda.cumath import sqrt as cusqrt
    from pycuda.cumath import atan
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as driver
    from pycuda.elementwise import ElementwiseKernel
    import cython
  

    # from pycuda.compiler import SourceModule as SM
    # from pycuda.cumath import sqrt as cusqrt
    
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
    sample1, l1 = create_dataset_base(samples = 1000, features = 30, centers = 2, display = False, n_dataset = 1)
    sample2, l2 = create_dataset_base(samples = 4000, features = 7, centers = 10, display = True, n_dataset = 2)
    sample3, l3 = create_dataset_base(samples = 7000, features = 30, centers = 10, display = True, n_dataset = 3)
    sample4, l4 = create_dataset_base(samples = 7000, features = 50, centers = 8, display = False, n_dataset = 4)
    sample5, l5 = create_dataset_base(samples = 10000, features = 1024, centers = 10, display = False, n_dataset = 5)
    sample6, l6 = create_dataset_base(samples = 15000, features = 5, centers = 7, display = False, n_dataset = 6)
    sample7, l7 = create_dataset_base(samples = 8000, features = 300, centers = 7, display = False, n_dataset = 7)   # -> !!!not working in cuda base 
    sample8, l8 = create_dataset_base(samples = 3, features = 4, centers = 2, display = False, n_dataset = 8)
    sample9, l9 = create_dataset_base(samples = 10000, features = 2048, centers = 5, display = False, n_dataset = 9)
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
    # listOfDataset.append((sample1, l1))
    # listOfDataset.append((sample2, l2))
    listOfDataset.append((sample3, l3))
    # listOfDataset.append((sample4, l4))
    listOfDataset.append((sample5, l5))
    # listOfDataset.append((sample6, l6))
    # listOfDataset.append((sample7, l7))
    # listOfDataset.append((sample8, l8))
    # listOfDataset.append((sample9, l9))
    # listOfDataset.append((sample10, l10))
    # listOfDataset.append((sample11, l11))

    # creation of the starting points -> theta  |  for each dataset
    PI = np.pi
    ''' listOfInitialTuple is a list of tuples '''
    listOfInitialTuple = []
    for eachDataset in listOfDataset:
        numberOfSamplesInTheDataset = eachDataset[0].shape[0]
        # new array of theta for the points inside each dataset
        theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
        listOfInitialTuple.append((eachDataset[0], eachDataset[1], theta))


    @timeit
    def get_data(thetaToTransform, normalize = True):
         # load the data
        data = np.copy(thetaToTransform[0])
        theta = np.copy(thetaToTransform[2])
        
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
        theta_sin = np.asarray(np.sin(theta) , np.float32)
        theta_cos = np.asarray(np.cos(theta) , np.float32)

        theta_gpu = gpuarray.to_gpu(theta)
        theta_sin_gpu = gpuarray.to_gpu(theta_sin)
        theta_cos_gpu = gpuarray.to_gpu(theta_cos)

        S_GPU = culinalg.dot(_weights_, theta_sin_gpu)
        C_GPU = culinalg.dot(_weights_, theta_cos_gpu)    

        return (_weights_.get(), C_GPU.get(), S_GPU.get(), theta_gpu.get(), _weights_, C_GPU, S_GPU, theta_gpu)

    @timeit
    def cpu_algorithm(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        start = time.time()
        matrixOfWeights = np.copy(matrixOfWeights_)
        C = np.copy(C_)
        S = np.copy(S_)
        theta = np.copy(theta_)
        end = time.time()
        
        ok = True
        rounds = 0
        thetaSize = theta.shape[0]

        while ok == True:
            ok = False
            rounds += 1
            nChanges = 0
            
            ''' loop on the theta '''
            for k in range(thetaSize):
                old = theta[k]  
                
                ''' find a theta that improves the equation '''
                theta[k] = np.arctan(S[k]/C[k])
                if C[k] >= 0:
                    theta[k] += PI
                elif S[k] > 0:
                    theta[k] += 2*PI

                # elementwise multiplication
                C += np.multiply(matrixOfWeights[k,:], np.repeat(np.cos(theta[k]) - np.cos(old), thetaSize))
                S += np.multiply(matrixOfWeights[k,:], np.repeat(np.sin(theta[k]) - np.sin(old), thetaSize))
                
                if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
                    ok = True
                    nChanges += 1
                
                '''
                print("old = {:.32f}".format(old))
                print("sin = {:.32f}".format(S[k]))
                print("cos = {:.32f}".format(C[k]))
                print("theta[k] = {:.32f}".format(theta[k]))
                # print("min = {:.32f}".format(min(abs(old - theta[k]), abs(2*PI - old + theta[k]))))
                print("np.cos(theta[k]) = {:.32f}".format(np.cos(theta[k])))
                print("np.sin(theta[k]) = {:.32f}".format(np.sin(theta[k])))
                print("val_sin = {:.32f}".format(np.sin(theta[k]) - np.sin(old)))
                print("val_cos = {:.32f}".format(np.cos(theta[k]) - np.cos(old)))
                '''

        return theta

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ################### methods ##############################################################
    ##########################################################################################

    ##########################################################################################
    # alg1
    @timeit
    def gpu_algorithm1(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        # todo -> prepare call 
        start = time.time()
        matrixOfWeights_ = np.float32(matrixOfWeights_)
        C_ = np.float32(C_)
        S_ = np.float32(S_)
        theta_ = np.float32(theta_)
        weights = gpuarray.to_gpu(matrixOfWeights_)
        C = gpuarray.to_gpu(C_)
        S = gpuarray.to_gpu(S_)
        theta = gpuarray.to_gpu(theta_)
        end = time.time()
        print(" time for copy = {0}".format(end - start))

        kernel = SM("""
            __global__ void elementwise_multiplication(float *weights, float* arr, float val, int thetaSize, int k)
            {
                int pos = threadIdx.x + blockDim.x * blockIdx.x;
                if (pos >= thetaSize)
                    return;

                int idx_weight = k * thetaSize + pos;
                arr[pos] += (weights[idx_weight] * val);
            }        
            """)

        block = (32, 1, 1)
        grid_x = (weights.shape[1] + (block[0] - 1)) // block[0]
        grid_y = 1
        grid = (grid_x, grid_y)

        # preparing kernel
        kernel_call = kernel.get_function("elementwise_multiplication")
        stream1 = driver.Stream()
        
        ok = True
        rounds = 0 
        thetaSize = theta.shape[0]
        thetaSize_int32 = np.int32(thetaSize)
        
        while ok == True:
            ok = False
            rounds += 1

            for k in range(thetaSize):
                old = theta[k].get().item()
                sin = S[k].get().item()
                cos = C[k].get().item()
                
                tmp = math.atan(sin/cos)
                
                if cos >= 0:
                    tmp += PI
                elif sin > 0:
                    tmp += 2*PI

                tmp_ = np.array([tmp]).astype(np.float32)    
                theta[k].set(tmp_) # -> todo: provare a rimuovere, forse collo di bottiglia 

                val_cos = math.cos(tmp) - math.cos(old)
                val_sin = math.sin(tmp) - math.sin(old)
                val_cos_float32 = np.float32(val_cos)
                val_sin_float32 = np.float32(val_sin)
                k_int32 = np.int32(k)

                # -> creare una sola chiamata  
                kernel_call(weights, C, val_cos_float32, thetaSize_int32, k_int32, block = block, grid = grid, stream = stream1)
                kernel_call(weights, S, val_sin_float32, thetaSize_int32, k_int32, block = block, grid = grid, stream = stream1)
                
                stream1.synchronize()

                if min(abs(old - tmp), abs(2*PI - old + tmp)) > eps:
                    ok = True

        return theta.get()

    # alg2
    @timeit
    def gpu_algorithm2(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        # todo -> everything in float32 instead of float64
        start = time.time()
        matrixOfWeights_ = np.float32(matrixOfWeights_)
        C_ = np.float32(C_)
        S_ = np.float32(S_)
        theta_ = np.float32(theta_)
        weights = gpuarray.to_gpu(matrixOfWeights_)
        C = gpuarray.to_gpu(C_)
        S = gpuarray.to_gpu(S_)
        theta = gpuarray.to_gpu(theta_)
        end = time.time()
        print(" time for copy = {0}".format(end - start))

        ok = True
        rounds = 0 
        thetaSize = theta.shape[0]
        
        while ok == True:
            ok = False
            rounds += 1

            for k in range(thetaSize):
                old = theta[k].get().item()
                sin = S[k].get().item()
                cos = C[k].get().item()
                
                tmp = math.atan(sin/cos)
                
                if cos >= 0:
                    tmp += PI
                elif sin > 0:
                    tmp += 2*PI

                tmp_ = np.array([tmp]).astype(np.float64)    
                theta[k].set(tmp_)

                val_cos = math.cos(tmp) - math.cos(old)
                val_sin = math.sin(tmp) - math.sin(old)


                # kernel_call(weights, C, val_cos_float32, thetaSize_int32, k_int32, block = block, grid = grid, stream = stream1)
                # kernel_call(weights, S, val_sin_float32, thetaSize_int32, k_int32, block = block, grid = grid, stream = stream1)

                # repeat_cos = gpuarray.to_gpu(np.repeat(np.cos(tmp) - np.cos(old), thetaSize))
                # repeat_sin = gpuarray.to_gpu(np.repeat(np.sin(tmp) - np.sin(old), thetaSize))

                val_cos_np = np.array([val_cos]).astype(np.float32)
                val_sin_np = np.array([val_sin]).astype(np.float32)
                val_cos_gpu = gpuarray.to_gpu(val_cos_np)
                val_sin_gpu = gpuarray.to_gpu(val_sin_np)
                
                C = misc.add(C, misc.multiply(weights[k,:], val_cos_gpu))
                S = misc.add(S, misc.multiply(weights[k,:], val_sin_gpu))

                if min(abs(old - tmp), abs(2*PI - old + tmp)) > eps:
                    ok = True

        return theta.get()


    # alg3
    @jit(nopython=True, parallel=True)
    def jit_elementwise_multiplication(line_weights, C, S, theta, k, old):
        # elementwise multiplication
        C += np.multiply(line_weights, np.repeat(np.cos(theta[k]) - np.cos(old), theta.shape[0]))
        S += np.multiply(line_weights, np.repeat(np.sin(theta[k]) - np.sin(old), theta.shape[0]))

    @timeit
    def jit_algorithm(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        matrixOfWeights = np.copy(matrixOfWeights_)
        C = np.copy(C_)
        S = np.copy(S_)
        theta = np.copy(theta_)
        
        ok = True
        rounds = 0
        thetaSize = theta.shape[0]

        while ok == True:
            ok = False
            rounds += 1
            nChanges = 0
            
            ''' loop on the theta '''
            for k in range(thetaSize):
                old = theta[k]
                
                ''' find a theta that improves the equation '''
                theta[k] = np.arctan(S[k]/C[k])
                if C[k] >= 0:
                    theta[k] += PI
                elif S[k] > 0:
                    theta[k] += 2*PI
                    
                jit_elementwise_multiplication(matrixOfWeights[k,:], C, S, theta, k, old)

                ''' exit condition '''
                if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
                    ok = True
                    nChanges += 1
                

        return theta


    # alg4
    @timeit
    def gpu_algorithm3(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        # todo -> prepare call 
        start = time.time()
        matrixOfWeights_ = np.float32(matrixOfWeights_)
        C_ = np.float32(C_)
        S_ = np.float32(S_)
        theta_ = np.float32(theta_)
        weights = gpuarray.to_gpu(matrixOfWeights_)
        C = gpuarray.to_gpu(C_)
        S = gpuarray.to_gpu(S_)
        theta = gpuarray.to_gpu(theta_)
        end = time.time()
        print(" time for copy = {0}".format(end - start))

        kernel = SM("""
            __global__ void elementwise_multiplication(float *weights, float *theta, float *C, float*S, float cos, float sin, float valueNewTheta, unsigned int thetaSize, unsigned int k)
            {
                int pos = threadIdx.x + blockDim.x * blockIdx.x; 
                if (pos >= thetaSize)
                    return;

                int idx_weight = k * thetaSize + pos;
                C[pos] += (weights[idx_weight] * cos);
                S[pos] += (weights[idx_weight] * sin);

                theta[k] = valueNewTheta;
            }        
            """)

        block = (32, 1, 1)
        grid_x = (weights.shape[1] + (block[0] - 1)) // block[0]
        grid_y = 1
        grid = (grid_x, grid_y)

        # preparing kernel
        kernel = kernel.get_function("elementwise_multiplication")
        kernel_call = kernel.prepare("PPPPfffii").prepared_call
        
        ok = True
        rounds = 0 
        thetaSize = theta.shape[0]
        thetaSize_int32 = np.uint32(thetaSize)
        
        while ok == True:
            ok = False
            rounds += 1

            for k in range(thetaSize):
                # mettere tutto quello che si puo sulla gpu 
                old = theta[k].get().item()
                sin = S[k].get().item()
                cos = C[k].get().item()
                
                tmp = math.atan(sin/cos)
                
                if cos >= 0:
                    tmp += PI
                elif sin > 0:
                    tmp += 2*PI
                
                val_cos = math.cos(tmp) - math.cos(old)
                val_sin = math.sin(tmp) - math.sin(old)
                val_cos_float32 = np.float32(val_cos)
                val_sin_float32 = np.float32(val_sin)
                val_new_theta = np.float32(tmp)
                k_int32 = np.uint32(k)

                kernel_call(grid, block, weights.gpudata,theta.gpudata, C.gpudata, S.gpudata, val_cos_float32, val_sin_float32, val_new_theta, thetaSize, k)

                if min(abs(old - tmp), abs(2*PI - old + tmp)) > eps:
                    ok = True

        return theta.get()


    # alg5
    # this method is affected by an big error approximation due to the computations done in the GPU
    @timeit
    def gpu_algorithm4(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        # todo -> prepare call 
        start = time.time()
        matrixOfWeights_ = np.float32(matrixOfWeights_)
        C_ = np.float32(C_)
        S_ = np.float32(S_)
        theta_ = np.float32(theta_)
        weights = gpuarray.to_gpu(matrixOfWeights_)
        C = gpuarray.to_gpu(C_)
        S = gpuarray.to_gpu(S_)
        theta = gpuarray.to_gpu(theta_)
        end = time.time()
        print(" time for copy = {0}".format(end - start))

        kernel = SM("""

            __device__ float mycos(float a){
                return cos(a);
            }

            __device__ float mysin(float a){
                return sin(a);
            }

            __global__ void elementwise_multiplication(float *weights, float *theta, float *C, float*S, float pi, unsigned int thetaSize, unsigned int k, int *ok, float eps)
            {
                int pos = threadIdx.x + blockDim.x * blockIdx.x; 
                if (pos >= thetaSize)
                    return;

                float old = theta[k];
                float sin = S[k];
                float cos = C[k];
                float tmp = atanf(sin/cos);

                if (cos >= 0)
                    tmp = tmp + pi;
                else if (sin > 0)
                    tmp = tmp + (2.0 * pi);

                float val_cos = mycos(tmp) - mycos(old);
                float val_sin = mysin(tmp) - mysin(old);

                int idx_weight = k * thetaSize + pos;
                C[pos] = C[pos] + (weights[idx_weight] * val_cos);
                S[pos] = S[pos] + (weights[idx_weight] * val_sin);
                
                if (pos == 0){
                    
                    theta[k] = tmp;

                    /*
                    printf("old = %.32f     "   , old);
                    printf("sin = %.32f     "   , sin);
                    printf("cos = %.32f     "   , cos);
                    printf("atanf(sin/cos) = %.32f     "   , atanf(sin/cos));
                    printf("theta[k] = %.32f    "   , theta[k]);
                    // printf("min = %.32f     "   , min(abs(old - tmp), abs(2*pi - old + tmp)));
                    printf("mycos(theta[k]) = %.32f     "   , mycos(theta[k]));
                    printf("mysin(theta[k]) = %.32f     "   , mysin(theta[k]));
                    printf("val_sin = %.32f    "   , val_sin);
                    printf("val_cos = %.32f    "   , val_cos);
                    */
                    
                    

                    if (    min(abs(old - tmp), abs(2*pi - old + tmp))   >   eps   ){
                        ok[pos] = 1;
                    }
                }
            }        

            """)

        block = (32, 1, 1)
        grid_x = (weights.shape[1] + (block[0] - 1)) // block[0]
        grid_y = 1
        grid = (grid_x, grid_y)

        # preparing kernel
        kernel = kernel.get_function("elementwise_multiplication")
        # kernel_call = kernel.prepare("PPPPPfii").prepared_call
        
        cont = True
        rounds = 0 

        thetaSize = theta.shape[0]
        thetaSize_int32 = np.uint32(thetaSize)
        pi_float32 = np.float32(np.pi)
        eps_float32 = np.float32(eps)
        
        while cont == True:
            cont = False
            rounds += 1

            for k in range(thetaSize):
                k_int32 = np.uint32(k)
                ok = np.zeros(shape = (1)).astype(np.int32)
                ok_gpu = gpuarray.to_gpu(ok)
        
                kernel(weights, theta, C, S, pi_float32, thetaSize_int32, k_int32, ok_gpu, eps_float32, grid = grid, block = block)
                driver.Context.synchronize()
                
                check = ok_gpu[0].get().item()

                if check == 1:
                    cont = True
                

        return theta.get()


    # alg6
    @jit(nopython=True, parallel=True)
    def jit_elementwise_multiplication2(line_weights, C, S, theta, k, old):
        # elementwise multiplication
        C += np.multiply(line_weights, np.repeat(np.cos(theta[k]) - np.cos(old), theta.shape[0]))
        S += np.multiply(line_weights, np.repeat(np.sin(theta[k]) - np.sin(old), theta.shape[0]))

    @jit(nopython=True) 
    def loop_jit(matrixOfWeights, theta, C, S, eps):
        
        ok = True
        rounds = 0
        thetaSize = theta.shape[0]

        while ok == True:
            ok = False
            rounds += 1
            nChanges = 0
            
            ''' loop on the theta '''
            for k in range(thetaSize):
                old = theta[k]  
                
                ''' find a theta that improves the equation '''
                theta[k] = np.arctan(S[k]/C[k])
                if C[k] >= 0:
                    theta[k] += PI
                elif S[k] > 0:
                    theta[k] += 2*PI
                    
                jit_elementwise_multiplication2(matrixOfWeights[k,:], C, S, theta, k, old)

                ''' exit condition '''
                if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
                    ok = True
                    nChanges += 1
        
        return theta
                
    @timeit
    def jit_algorithm2(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        matrixOfWeights = np.copy(matrixOfWeights_)
        C = np.copy(C_)
        S = np.copy(S_)
        theta = np.copy(theta_)
        
        return loop_jit(matrixOfWeights, theta, C, S, eps)

    # alg7
    @jit(nopython=True, parallel=True)
    def jit_elementwise_multiplication3(line_weights, C, S, theta, k, old):
        # elementwise multiplication
        C += np.multiply(line_weights, np.repeat(np.cos(theta[k]) - np.cos(old), theta.shape[0]))
        S += np.multiply(line_weights, np.repeat(np.sin(theta[k]) - np.sin(old), theta.shape[0]))

    @jit(nopython=True) 
    def loop_jit_cython(matrixOfWeights : cython.float, theta : cython.float, C : cython.float, S : cython.float, eps : cython.float):
        
        ok : cython.bool
        rounds : cython.int
        
        ok = True
        rounds = 0
        thetaSize : cython.int
        thetaSize = theta.shape[0]

        while ok == True:
            ok = False
            rounds += 1
            
            for k in range(thetaSize):
                old : cython.float
                old = theta[k]
                
                ''' find a theta that improves the equation '''
                theta[k] = np.arctan(S[k]/C[k])
                if C[k] >= 0:
                    theta[k] += PI
                elif S[k] > 0:
                    theta[k] += 2*PI
                    
                jit_elementwise_multiplication2(matrixOfWeights[k,:], C, S, theta, k, old)

                if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
                    ok = True
        
        return theta
                
    @timeit
    def jit_cython_algorithm(matrixOfWeights_, C_, S_, theta_,  eps = 0.01):
        
        matrixOfWeights = cython.float[matrixOfWeights_.shape[0] * matrixOfWeights_.shape[1]]
        matrixOfWeights = np.copy(matrixOfWeights_)
        C = cython.float[C_.shape[0]]
        C = np.copy(C_)
        S = cython.float[S_.shape[0]]
        S = np.copy(S_)
        theta = cython.float[theta_.shape[0]]
        theta = np.copy(theta_)
        
        return loop_jit_cython(matrixOfWeights, theta, C, S, eps)

    
    # process the data
    counter = 0
    for eachTupleBeforeTransformation in listOfInitialTuple:
        counter += 1

        print("////////////////DATABASE {0}////////////////".format(counter))

        print("get_data ---------------------------------------------------------")
        weights_cpu, C_cpu, S_cpu, theta_cpu, weights_gpu, C_gpu, S_gpu, theta_gpu = get_data(thetaToTransform = eachTupleBeforeTransformation, normalize = True)

        print("cpu _ version ----------------------------------------------------")
        theta_cpu_computed = cpu_algorithm(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        
        '''
        print("gpu _ version 1 --------------------------------------------------")
        theta_gpu_computed1 = gpu_algorithm1(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        print(checkResults(theta_cpu_computed, theta_gpu_computed1))

        # print("gpu _ version 2 ------------------------------------------------")
        # theta_gpu_computed2 = gpu_algorithm2(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        # print(checkResults(theta_cpu_computed, theta_gpu_computed2))
        '''

        print("jit _ version 3 --------------------------------------------------")
        theta_jit_computed3 = jit_algorithm(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        print(checkResults(theta_cpu_computed, theta_jit_computed3))

        
        print("gpu _ version 4 --------------------------------------------------")
        theta_gpu_computed3 = gpu_algorithm3(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        print(checkResults(theta_cpu_computed, theta_gpu_computed3))
        assert_almost_equal(theta_cpu_computed, theta_gpu_computed3)
        
        '''
        print("gpu _ version 5 --------------------------------------------------")
        theta_gpu_computed4 = gpu_algorithm4(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        print(checkResults(theta_cpu_computed, theta_gpu_computed4))
        '''

        print("jit _ version 6 --------------------------------------------------")
        theta_jit_computed5 = jit_algorithm2(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        print(checkResults(theta_cpu_computed, theta_jit_computed5))

        print("jit + cython _ version 7 --------------------------------------------------")
        theta_jit_cython_computed6 = jit_cython_algorithm(matrixOfWeights_ = weights_cpu, C_ = C_cpu, S_ = S_cpu, theta_ = theta_cpu, eps = 0.001)
        print(checkResults(theta_cpu_computed, theta_jit_cython_computed6))

        
        