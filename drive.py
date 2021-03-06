# -*- coding: utf-8 -*-
"""Circle_Clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S6gJjNr6Z1ViIZq8-G6N9qkNhlXc-jWe

# **`CLUSTERING ALGORITHM ON THE UNIT CIRCLE `**

The following page shows the implementations of a new clustering method based on the unit circle. 

First the single core version of the algorithm is implemented then the optimized versions. 

For each implementation, the version, the libraries used and the time/memory consumed are indicated.

The algorithm is tested using datasets of different size and different dimensions
"""

import math
import cupy as cp
from numba import jit, cuda
from sklearn.metrics.pairwise import euclidean_distances
# ignore warnings for jit
import warnings
warnings.filterwarnings("ignore")

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

def pairwise_euclidean_distance_multithread(X, Y, size, start, stop, finalMat):
  uploadMatrix = X[start:stop, 0:size]
  M = euclidean_distances(uploadMatrix, Y)
  row = M.shape[0]
  col = M.shape[1]
  for i in range(row):
    for j in range(col):
      finalMat[i + start, j] = M[i, j]

# check if the results are correct
def checkResults(array1, array2):
  for i in range(array1.shape[0]):
    if abs(array1[i] - array2[i]) > 0.0001:
      return False
  return True

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
  
  n_cores = multiprocessing.cpu_count()
  print('cores = {0}'.format(n_cores))
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

  # sample0, l0 = create_dataset_iris(display = False, n_dataset = 0)
  #??sample1, l1 = create_dataset_base(samples = 1000, features = 3, centers = 2, display = False, n_dataset = 1)
  # sample2, l2 = create_dataset_base(samples = 5000, features = 7, centers = 10, display = True, n_dataset = 2)
  #??sample3, l3 = create_dataset_base(samples = 7000, features = 30, centers = 10, display = True, n_dataset = 3)
  sample4, l4 = create_dataset_base(samples = 5000, features = 50, centers = 8, display = True, n_dataset = 4)
  # sample5, l5 = create_dataset_base(samples = 10000, features = 20, centers = 10, display = True, n_dataset = 5)
  sample6, l6 = create_dataset_base(samples = 3000, features = 1000, centers = 7, display = True, n_dataset = 6)
  # sample4, l4 = create_dataset_moon(samples = 1000, noise = 0.05, display = True, n_dataset = 4)
  # sample5, l5 = create_dataset_moon(samples = 1000, noise = 0.2, display = True, n_dataset = 5)
  # sample6, l6 = create_dataset_circles(samples = 1000, noise = 0.05, display = True, n_dataset = 6)
  # sample7, l7 = create_dataset_classification(n_samples = 100, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = True, n_dataset = 7)
  # sample8, l8 = create_dataset_classification(n_samples = 1000, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = True, n_dataset = 8)
  #??sample9, l9 = create_dataset_olivetti_faces(display = True, n_dataset = 9)

  listOfDataset = []
  #??listOfDataset.append((sample0, l0))
  #??listOfDataset.append((sample1, l1))
  #??listOfDataset.append((sample2, l2))
  #??listOfDataset.append((sample3, l3))
  listOfDataset.append((sample4, l4))
  #??listOfDataset.append((sample5, l5))
  listOfDataset.append((sample6, l6))
  # listOfDataset.append((sample7, l7))
  # listOfDataset.append((sample8, l8))
  #??listOfDataset.append((sample9, l9))

  # creation of the starting points -> theta  |  for each dataset
  PI = np.pi
  ''' listOfInitialTuple is a list of tuples '''
  listOfInitialTuple = []
  for eachDataset in listOfDataset:
    numberOfSamplesInTheDataset = eachDataset[0].shape[0]
    # new array of theta for the points inside each dataset
    theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
    listOfInitialTuple.append((eachDataset[0], eachDataset[1], theta))

  ''' each tuple contains three elements --> the list of samples, the class of each sample, the starting randomic thetas'''

  """### **Version** : 1.0

  ***Libraries used*** : Numpy

  ***Description*** : This is most obvious way to execute the algorithm. The euclidean distance is done simply using the dedicated function in sklearn.metrics.pairwise. The algorithm uses the libraries of numpy like np.dot(), np.multiply and np.repeat()
  """

  def circleClustering(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }
    init = time.time()
    data = np.copy(thetaToTransform[0])
    matrixOfWeights = euclidean_distances(data, data)
    theta = np.copy(thetaToTransform[2])
    if normalize:
      matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    
    end = time.time()
    print("time = {0}".format(end - init))
    
    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    S = np.dot(matrixOfWeights, sin_t)

    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]
    while ok:
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

        ''' exit condition '''
        if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
          ok = True
          nChanges += 1

    return theta
    
  '''
  # process the data and print plot the results
  listOfProcessedTuple = []
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTuple.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  list_results.append(("1.1.0", listOfProcessedTuple))
  '''
  
  # plotting the data
  def plot_circle(theta, label=None, radius=500):
    """
      Produce a plot with the locations of all poles and zeros
    """

    x = np.cos(theta)
    y = np.sin(theta)

    fig = go.Figure()
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="black", width=1))
    
    if label is None:
      fig.add_trace(go.Scatter(x=x, y=y,
            mode='markers',
            marker_symbol='circle',
            marker_size=10))
    else:
      ul = np.unique(label)
      cols = list(range(len(ul)))
      for c,u in zip(cols,ul):
        idx = np.where(u == label)
        fig.add_trace(go.Scatter(x=x[idx], y=y[idx],
            mode='markers',
            marker_symbol='circle',
            marker_color=cols[c], 
            marker_line_color=cols[c],
            marker_line_width=0, 
            marker_size=10))
    
    M = 1.05
    fig.update_xaxes(title='', range=[-M, M])
    fig.update_yaxes(title='', range=[-M, M])
    fig.update_layout(title='clusters', width=radius, height=radius)
    fig.show()


  # for data, target, processedTheta, times in listOfProcessedTuple:
    # plot_circle(theta = processedTheta, label = target)

  """
  ## PARALLELIZATION

  The two loops inside the algorithm can't be parallelized because each one has computations that depends from the calculus of the previous interation.

  Let's take a look to this line of the for loop.
  
  ***theta[k] = np.arctan(S[k] / C[k])***

  Here the values of *S[k]* and *C[k]* are obtained from the previous iteration so this part can't be executed in parallel.

  Also the external while loop can't take advantage of the multithread paradigm because the decision of when to stop happens inside the for loop and the arctan function problem repeats itself again.

  For parallelizing this algorithm it's possible to operate just on the first and second part where are computed the matrix of weights(parallelization of the euclidean distance) and the calculus of the cosines and sines. 

  Also it's possible to parallelize the re-computing of C[] and S[] inside the internal loop



  1.   Euclidean Distance
  2.   sine and cosine summations
  3.   Re-computation of C[] and S[]
  """
  
  """
  def circleClustering_1_1_1(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }
    ''' dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y)) '''
    
    # matrixOfWeights = euclidean_distances(thetaToTransform[0], thetaToTransform[0])
    X = thetaToTransform[0]
    size = thetaToTransform[0].shape[0]
    matrixOfWeights = np.empty([size, size])

    _size_ = size // n_cores
    listOfThreads = []
    for i in range(n_cores):
      if i < n_cores-1:
        t = threading.Thread(target = loop_euclidean_distance, args = (X, size, i*_size_, (i+1)*_size_, matrixOfWeights, ))
      else:
        t = threading.Thread(target = loop_euclidean_distance, args = (X, size, i*_size_, size, matrixOfWeights, ))
      listOfThreads.append(t)

    for t in listOfThreads:
      t.start()
    for t in listOfThreads:
      t.join()

      
    
    theta = thetaToTransform[2]
    if normalize:
      matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    


    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    S = np.dot(matrixOfWeights, sin_t)

    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]
    while ok:
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
        
        C += np.multiply(matrixOfWeights[k,:], np.repeat(np.cos(theta[k]) - np.cos(old), thetaSize))
        S += np.multiply(matrixOfWeights[k,:], np.repeat(np.sin(theta[k]) - np.sin(old), thetaSize))

        ''' exit condition '''
        if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
          ok = True
          nChanges += 1

    return theta
  
  # process the data 
  listOfProcessedTupleVersion1_1_1 = []
  print("////// version 1.1.1")
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_1_1(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTupleVersion1_1_1.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  for i in range(len(listOfProcessedTuple)):
    a = listOfProcessedTuple[i]
    b = listOfProcessedTupleVersion1_1_1[i]
    print('test dataset {0} = {1}'.format(i, checkResults(a[2], b[2])))
  
  list_results.append(("1.1.1", listOfProcessedTupleVersion1_1_1))

  """

  ### **Version** : 1.1.2

  # ***Libraries used*** : None

  #??*Description* : This implementation makes use of the multithreading on the CPU. But this time it is used the multiprocessing library

  """
  def circleClustering_1_1_2(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }
    ''' dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y)) '''
    
    # matrixOfWeights = euclidean_distances(thetaToTransform[0], thetaToTransform[0])
    X = thetaToTransform[0]
    size = thetaToTransform[0].shape[0]
    matrixOfWeights = np.empty([size, size])

    _size_ = size // n_cores
    listOfThreads = []
    for i in range(n_cores):
      if i < n_cores-1:
        t = multiprocessing.Process(target = loop_euclidean_distance, args = (X, size, i*_size_, (i+1)*_size_, matrixOfWeights, ))
      else:
        t = multiprocessing.Process(target = loop_euclidean_distance, args = (X, size, i*_size_, size, matrixOfWeights, ))
      listOfThreads.append(t)

    for t in listOfThreads:
      t.start()
      
    for t in listOfThreads:
      t.join()


    #??loop_euclidean_distance(X, size, 0, size, matrixOfWeights)
    
    theta = thetaToTransform[2]
    if normalize:
      matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    


    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    S = np.dot(matrixOfWeights, sin_t)

    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]
    while ok:
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
        
        C += np.multiply(matrixOfWeights[k,:], np.repeat(np.cos(theta[k]) - np.cos(old), thetaSize))
        S += np.multiply(matrixOfWeights[k,:], np.repeat(np.sin(theta[k]) - np.sin(old), thetaSize))

        ''' exit condition '''
        if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
          ok = True
          nChanges += 1

    return theta

  # process the data 
  listOfProcessedTupleVersion1_1_2 = []
  print("////// version 1.1.2")
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_1_2(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTupleVersion1_1_2.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  for i in range(len(listOfProcessedTuple)):
    a = listOfProcessedTuple[i]
    b = listOfProcessedTupleVersion1_1_2[i]
    print('test dataset {0} = {1}'.format(i, checkResults(a[2], b[2])))
  
  list_results.append(("1.1.2", listOfProcessedTupleVersion1_1_2))
  """

  ### **Version** : 1.1.3

  #??**Libraries used** : Numpy

  # Description : This implementation makes use of the multithreading on the CPU. 
  #??The computation is done using the euclidean_distance from sklearn, but the results is computed dividing the initial matrix in n_cores different matrices

  
  def circleClustering_1_1_3(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }
    
    # matrixOfWeights = euclidean_distances(thetaToTransform[0], thetaToTransform[0])
    init = time.time()
    X = np.copy(thetaToTransform[0])
    size = thetaToTransform[0].shape[0]
    matrixOfWeights = np.empty([size, size])

    _size_ = size // n_cores
    listOfThreads = []
    for i in range(n_cores):
      if i < n_cores-1:
        t = multiprocessing.Process(target = pairwise_euclidean_distance_multithread, args = (X, X, size, i*_size_, (i+1)*_size_, matrixOfWeights, ))
      else:
        t = multiprocessing.Process(target = pairwise_euclidean_distance_multithread, args = (X, X, size, i*_size_, (i+1)*_size_, matrixOfWeights, ))
      listOfThreads.append(t)

    for t in listOfThreads:
      t.start()
    
    for t in listOfThreads:
      t.join()
      
    
    theta = np.copy(thetaToTransform[2])
    if normalize:
      matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    

    end = time.time()
    print("time = {0}".format(end - init))

    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    S = np.dot(matrixOfWeights, sin_t)

    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]
    while ok:
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

        ''' exit condition '''
        if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
          ok = True
          nChanges += 1

    return theta

  # process the data and print plot the results
  listOfProcessedTuple = []
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_1_3(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTuple.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  list_results.append(("1.1.3", listOfProcessedTuple))
  ''' Those are the times of execution for each dataset'''

  '''
  # process the data 
  listOfProcessedTupleVersion1_1_3 = []
  print("////// version 1.1.3")
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_1_3(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTupleVersion1_1_3.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  for i in range(len(listOfProcessedTuple)):
    a = listOfProcessedTuple[i]
    b = listOfProcessedTupleVersion1_1_3[i]
    print('test dataset {0} = {1}'.format(i, checkResults(a[2], b[2])))
  
  list_results.append(("1.1.3", listOfProcessedTupleVersion1_1_3))
  '''
  
  
  ### **Version** : 1.1.4

  #??**Libraries used** : Numpy, Numba

  # Description : The computation is done using the euclidean_distance from sklearn without multithreading. It is used the just in time compiler of Numba. The function produces a lot of warnings because unable to use the noPython functionality.
  # Indeed jit can work in python mode or in Object mode. If is not able to use python mode (the fast one), he automatically switches in Object mode. 
  # The improvement is not very high.
  

  @jit
  def circleClustering_1_1_4(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }
    data = np.copy(thetaToTransform[0])
    matrixOfWeights = euclidean_distances(data, data)
    theta = np.copy(thetaToTransform[2])
    if normalize:
      matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    


    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    S = np.dot(matrixOfWeights, sin_t)

    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]
    while ok:
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

        ''' exit condition '''
        if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
          ok = True
          nChanges += 1

    return theta

  # process the data 
  listOfProcessedTupleVersion1_1_4 = []
  print("////// version 1.1.4")
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_1_4(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTupleVersion1_1_4.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  for i in range(len(listOfProcessedTuple)):
    a = listOfProcessedTuple[i]
    b = listOfProcessedTupleVersion1_1_4[i]
    print('test dataset {0} = {1}'.format(i, checkResults(a[2], b[2])))
  
  list_results.append(("1.1.4", listOfProcessedTupleVersion1_1_4))



  ### **Version** : 1.1.5

  #??**Libraries used** : Numpy, Numba

  # Description : The computation is done computing manually the euclidean distance using the upper functions. It is used the just in time compiler of Numba only on the two functions. 

  def circleClustering_1_1_5(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }

    X = np.copy(thetaToTransform[0])
    size = thetaToTransform[0].shape[0]
    matrixOfWeights = np.empty([size, size])

    loop_euclidean_distance_jit(X, size, 0, size, matrixOfWeights)

    theta = np.copy(thetaToTransform[2])
    if normalize:
      matrixOfWeights = matrixOfWeights / linalg.norm(matrixOfWeights)    


    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    S = np.dot(matrixOfWeights, sin_t)

    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]
    while ok:
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

        ''' exit condition '''
        if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
          ok = True
          nChanges += 1

    return theta

  # process the data 
  listOfProcessedTupleVersion1_1_5 = []
  print("////// version 1.1.5")
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_1_5(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTupleVersion1_1_5.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  for i in range(len(listOfProcessedTuple)):
    a = listOfProcessedTuple[i]
    b = listOfProcessedTupleVersion1_1_5[i]
    print('test dataset {0} = {1}'.format(i, checkResults(a[2], b[2])))
  
  list_results.append(("1.1.5", listOfProcessedTupleVersion1_1_5))

  """
  ### **At this point, the best result was simply adding the @jit decorator.**
  Now we will begin to extend the techniques also on the other parts of the code that can be parallelized using the GPU as well.

  ### **Version** : 1.2.1

  **Libraries used** : Numpy, Numba, Cuda(Numba)

  Description : All three parts are transferred to GPU, in particular we will try to make parallel the calculation of the Euclidean distance, the calculation of the initial vectors C[] and S [], and the final recalculation: 

  *C += np.multiply(...)*

  *S += np.multiply(...)*
  """

  '''
  def checkResultsGPU(array1, array2):
    for i in range(array1.shape[0]):
      if abs(array1[i] - array2[i]) > 0.0001:
        return False
    return True

  def compare(arrayCP, arrayGPU):
    arrayCPGPU = cp.array(arrayCP)
    return checkResultsGPU(arrayCPGPU, arrayGPU)

  def checkResultsTableGPU(array1, array2):
    for i in range(array1.shape[0]):
      for j in range(array1.shape[1]):
        if abs(array1[i,j] - array2[i,j]) > 0.0001:
          return False
    return True

  def compareTable(matCP, matGPU):
    matCPGPU = cp.array(matCP)
    return checkResultsTableGPU(matCPGPU, matGPU)

  '''

  def circleClustering_1_2_1(thetaToTransform, eps = 0.01, normalize = True):
    # PART 1 -> {
        # matrixOfWeights | get the matrix of weight using a formula of distance between points in n dimension   
        # theta           | get the theta

        # normalization of the matrix of weights with Frobenius norm
          # -> sqrt ( summation |a(i,j|^2 )
    # }

    X = np.copy(thetaToTransform[0])
    size = thetaToTransform[0].shape[0]
    
    matrixOfWeights = euclidean_distances(X, X)
    theta = np.copy(thetaToTransform[2])

    # transform to cupy array
    matrixOfWeightsGPU = cp.array(matrixOfWeights)
    thetaGPU = cp.array(theta)
    
    
    if normalize:
      matrixOfWeightsGPU = matrixOfWeightsGPU / cp.linalg.norm(matrixOfWeightsGPU) 
    

    # PART 2    -> computing  (summation of weights[i][j] * cos theta[j]) 
    #           -> computing  (summation of weights[i][j] * sin theta[j])

    sin_tGPU = cp.sin(thetaGPU)
    cos_tGPU = cp.cos(thetaGPU)
    C_GPU = cp.dot(matrixOfWeightsGPU, cos_tGPU)
    S_GPU = cp.dot(matrixOfWeightsGPU, sin_tGPU)


    # PART 3    -> cycle until we are below the eps threshold
    ok = True
    rounds = 0
    thetaSizeGPU = thetaGPU.shape[0]

    
    while ok:
      ok = False
      rounds += 1
      nChanges = 0
      
      ''' loop on the theta '''
      for k in range(thetaSizeGPU):
        oldGPU = thetaGPU[k].item()

        thetaGPU[k] = cp.arctan(S_GPU[k]/C_GPU[k])
        if C_GPU[k] >= 0:
          thetaGPU[k] += PI
        elif S_GPU[k] > 0:
          thetaGPU[k] += 2*PI

        # elementwise multiplication GPU
        C_GPU += cp.multiply(matrixOfWeightsGPU[k,:], cp.repeat(cp.cos(thetaGPU[k]) - cp.cos(oldGPU), thetaSizeGPU))
        S_GPU += cp.multiply(matrixOfWeightsGPU[k,:], cp.repeat(cp.sin(thetaGPU[k]) - cp.sin(oldGPU), thetaSizeGPU))

        ''' exit condition '''
        if min(abs(oldGPU - thetaGPU[k]), abs(2*PI - oldGPU + thetaGPU[k])) > eps:
          ok = True
          nChanges += 1

    return thetaGPU

  # process the data 
  listOfProcessedTupleVersion1_2_1 = []
  print("////// version 1.2.1")
  for eachTupleBeforeTransformation in listOfInitialTuple:
    ts = time.time()
    newThetas = circleClustering_1_2_1(thetaToTransform = eachTupleBeforeTransformation, eps = 0.001)
    te = time.time()
    print('%2.2f sec' % \
                (te-ts))
    listOfProcessedTupleVersion1_2_1.append((
        eachTupleBeforeTransformation[0],
        eachTupleBeforeTransformation[1],
        newThetas,
        te-ts
    ))

  for i in range(len(listOfProcessedTuple)):
    a = listOfProcessedTuple[i]
    b = listOfProcessedTupleVersion1_2_1[i]
    '''
    print("------------------------------")
    print("a = {}".format(a[2]))
    print("b = {}".format(b[2]))
    '''
    check = checkResults(a[2], b[2])
    print('test dataset {0} = {1}'.format(i, check))
  
  list_results.append(("1.2.1", listOfProcessedTupleVersion1_2_1))



  print('RESULTS')

  for i in range(len(list_results)):
    obj = list_results[i]
    print("Version : {0}".format(obj[0]))
    data = obj[1]
    count = 0
    for eachRun in data:
      print("Run {0} , time : {1}".format(count, eachRun[3]))
      count += 1