import time
import numpy as np
from numpy.random import rand
from numpy import copy
from numpy.testing import assert_almost_equal
import cclustering
import cclustering_cpu as cc_cpu
import cclustering_gpu as cc_gpu
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pycuda.gpuarray as gpuarray
import utils


PI = np.pi
PI = np.float32(PI)

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

def scaling(samples):
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)
    return samples

def create_dataset_base(samples, features, centers, standard_deviation_cluster = 1, standard = True, display = False, n_dataset = 0):
    # X = The generated samples
    # l = The integer labels for cluster membership of each sample
    X, l = make_blobs(n_samples = samples, n_features = features, centers = centers, cluster_std = standard_deviation_cluster, random_state = None)
    if standard:
        X = scaling(X)
    if display: 
        doPCA(X, l, n_dataset)
    return X, l

sample0, l0 = create_dataset_base(samples = 5, features = 4, centers = 2, display = False, n_dataset = 0)
sample1, l1 = create_dataset_base(samples = 1000, features = 30, centers = 2, display = False, n_dataset = 1)
sample2, l2 = create_dataset_base(samples = 4000, features = 7, centers = 10, display = False, n_dataset = 2)
sample3, l3 = create_dataset_base(samples = 7000, features = 30, centers = 10, display = False, n_dataset = 3)
sample4, l4 = create_dataset_base(samples = 10000, features = 1024, centers = 5, display = False, n_dataset = 4)
sample5, l5 = create_dataset_base(samples = 13000, features = 1024, centers = 5, display = False, n_dataset = 5)
sample6, l6 = create_dataset_base(samples = 15000, features = 2048, centers = 5, display = False, n_dataset = 6)
sample7, l7 = create_dataset_base(samples = 17000, features = 2048, centers = 8, display = False, n_dataset = 7)

listOfDataset = []
listOfDataset.append((sample0, l0))
listOfDataset.append((sample1, l1))
listOfDataset.append((sample2, l2))
listOfDataset.append((sample3, l3))
# listOfDataset.append((sample4, l4))
# listOfDataset.append((sample5, l5))
# listOfDataset.append((sample6, l6))
# listOfDataset.append((sample7, l7))


def checkResultsArray(array1, array2, error):
    '''
    if array1.shape[0] != array2.shape[0]:
        return False
    '''
    for i in range(array1.shape[0]):
        if abs(abs(array1[i]) - abs(array2[i])) > error:
            # if abs(array1[i, j]) - abs(array2[i, j]) != 0:
            print(abs(array1[i]) - abs(array2[i]))
            print("index = ", i)
            return False
    return True

def checkResultsMatrix(array1, array2):
    '''
    if array1.shape[0] != array2.shape[0]:
        return False
    '''
    sizeMatrix = array1.shape[0]
    for i in range(sizeMatrix):
        for j in range(sizeMatrix):
            if abs(abs(array1[i, j]) - abs(array2[i, j])) > 0.001 :
                print(abs(abs(array1[i, j]) - abs(array2[i, j])))
                print("index = i:{0} j:{1}".format(i,j))
                return False
    return True

def checkResultsMatrixWeights(array1, array2, error):
    '''
    if array1.shape[0] != array2.shape[0]:
        return False
    '''
    sizeMatrix = array1.shape[0]
    for i in range(sizeMatrix):
        for j in range(sizeMatrix):
            if abs(abs(array1[i, j]) - abs(array2[i, j])) > error :
                print(abs(abs(array1[i, j]) - abs(array2[i, j])))
                print("index = i:{0} j:{1}".format(i,j))
                return False
    return True

def checkResultsMatrixPrecise(matrix1, matrix2):
    row = matrix1.shape[0]
    col = matrix2.shape[1]
    for i in range(row):
        for j in range(col):
            if matrix1[i, j] != matrix2[i, j]:
                print("difference == 1) {0}    2) {1}".format(matrix1[i, j], matrix2[i, j]))
                return False
    return True

def printStringDatabase(c):
    print("////////////////DATABASE {0}////////////////".format(c))

def printStringStarted(t, p):
    print(" >>>> test nr. {0} ... precision = {1} <<<<".format(t, p))

def printStringTestPassed():
    print(" ---- passed ----")

def printStringTestPassedWeights():
    print(" ---- passed weights ----")

def printStringTestPassed_S_C():
    print(" ---- passed S C ----")

def printStringTestPassedLoop():
    print(" ---- passed Loop ----")

def printStringTestLoopFailed():
    print(" ---- Loop Failed ----")

def testDB(tuple_, precision, testnr):
    
    printStringStarted(t = testnr, p = precision)
    
    dataset = copy(tuple_[0])
    valuesCPU = copy(dataset)
    valuesGPU = copy(dataset)
    
    theta = 2 * PI * rand(tuple_[0].shape[0])
    theta_CPU = copy(theta)
    theta_GPU = copy(theta)
    
    weightscpu, Scpu, Ccpu, thetacpu, targetcpu = cclustering.CircleClustering(dataset = valuesCPU, precision = precision, hardware = "cpu", _theta_ = theta_CPU)
    weightsgpu, Sgpu, Cgpu ,thetagpu, targetgpu = cclustering.CircleClustering(dataset = valuesGPU, precision = precision, hardware = "gpu", _theta_ = theta_GPU)
    
    print("check weights")
    assert_almost_equal(weightscpu, weightsgpu.get())
    print("check S")
    assert checkResultsArray(Scpu, Sgpu.get(), 0.001)
    print("check C")
    assert checkResultsArray(Ccpu, Cgpu.get(), 0.001)
    
    if checkResultsArray(thetacpu, thetagpu, 0.001):
        printStringTestPassed()
    else:
        printStringTestLoopFailed()


"""
    test the three full versions of the algorithm
"""
counter = 0
for eachTuple in listOfDataset:
    
    counter += 1    
    
    '''
    printStringDatabase(c = counter)
    
    testDB(tuple_ = eachTuple, precision = "low", testnr = 1)
    testDB(tuple_ = eachTuple, precision = "medium", testnr = 2)
    # testDB(tuple_ = eachTuple, precision = "high", testnr = 3)
    
    '''



# test the three versions of the algorithm in separated parts
counter = 0
for eachTuple in listOfDataset:


    counter += 1
    printStringDatabase(c = counter)
    
    dataset = copy(eachTuple[0])
    valuesCPU = copy(dataset)
    valuesGPU = copy(dataset)
    
    theta = 2 * PI * rand(eachTuple[0].shape[0])
    theta_CPU = copy(theta)
    theta_GPU = copy(theta)

    # -> computing data
    print("computing cpu weights")
    start = time.time()
    weights_CPU, S_CPU, C_CPU = cc_cpu.computing_weights(valuesCPU, theta_CPU)
    end = time.time()
    print(" >>> time cpu = {0}".format(end - start))
    
    print("computing gpu weights")
    start = time.time()
    weights_GPU, S_GPU, C_GPU = cc_gpu.computing_weights(valuesGPU, theta_GPU)
    end = time.time()
    print(" >>> time gpu = {0}".format(end - start))

    weights_GPU_converted, S_GPU_converted, C_GPU_converted = utils.getDataFromGpu(weights_GPU, S_GPU, C_GPU)
    # weights_GPU_converted_main, S_GPU_converted_main, C_GPU_converted_main = utils.getDataFromGpu(weights_GPU_main, S_GPU_main, C_GPU_main)
    
   
    print("check weights")
    assert_almost_equal(weights_CPU, weights_GPU_converted)
    print("check S")
    assert_almost_equal(S_CPU, S_GPU_converted)
    print("check C")
    assert_almost_equal(C_CPU, C_GPU_converted)

    '''
    print("computing the loop in cpu")
    start = time.time()
    theta_CPU = cc_cpu.loop(weights_CPU, theta_CPU, S_CPU, C_CPU, 0.001)
    end = time.time()
    print(" >>> time cpu = {0}".format(end - start))

    print("computing the loop in gpu")
    start = time.time()
    theta_GPU = cc_gpu.loop_gpu(weights_GPU, theta_GPU, S_GPU, C_GPU, 0.001)
    end = time.time()
    print(" >>> time cpu = {0}".format(end - start))

    print("check loop values")
    if checkResultsArray(theta_CPU, theta_GPU, 0.001):
        print("TEST PASSED")
    else:
        print("TEST FAILED")

    '''    


# check the incorrect inputs
'''
    input is CircleClustering(dataset, target, precision, hardware, _theta_)
'''
print("*********************")
print("******testinputs*****")
print("*********************")

print("++++++test nr 1++++++")
print("check dataset")
datas = [] 
assert cclustering.CircleClustering(dataset = datas) == 1
datas = np.empty([2])
assert cclustering.CircleClustering(dataset = datas) == 1
datas = np.array([[2.56, 2,14]])
assert cclustering.CircleClustering(dataset = datas) == 1

print("++++++test nr 2++++++")
print("check dataset")
datas = np.copy(listOfDataset[0][0])
prec = "ggg"
assert cclustering.CircleClustering(dataset = datas, precision = prec) == 2

print("++++++test nr 3++++++")
print("check dataset")
datas = np.copy(listOfDataset[0][0])
prec = "medium"
hard = "c"
assert cclustering.CircleClustering(dataset = datas, precision = prec, hardware = hard) == 3