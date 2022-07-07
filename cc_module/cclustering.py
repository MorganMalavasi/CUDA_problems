import numpy as np
import cclustering_cpu as cc_cpu
import cclustering_gpu as cc_gpu
import utils

from numba import cuda

# constants
PI = np.pi
PI = np.float32(PI)

# create only one exception
class Circleclustering_input_zero(Exception):
    # Constructor method
    def __init__(self, value):
        self.value = value
    # __str__ display function
    def __str__(self):
        return(repr(self.value))
    
class Circleclustering_wrong_precision_string(Exception):
    # Constructor method
    def __init__(self, value):
        self.value = value
    # __str__ display function
    def __str__(self):
        return(repr(self.value))

class Circleclustering_hardware_wrong(Exception):
    # Constructor method
    def __init__(self, value):
        self.value = value
    # __str__ display function
    def __str__(self):
        return(repr(self.value))
    

def CircleClustering(dataset, target = None, precision = None, hardware = None, _theta_ = None):
    
    """
    Compute the points on the circle representing the dataset

    Parameters
    ----------
    
    dataset : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
        The number of columns is the number of features (dimension)
    
    target  : ndarray (optional)
        1D Numpy array of int dtype representing the class of each point in
        the dataset
    
    precision     : string (optional)
        Achievable error threshold.
        3 types ->  "high"    (0.0001)
                ->  "medium"  (0.001)
                ->  "low"     (0.01)

    hardware    : string (optional)
        Specific hardware to use
        3 types -> "cpu"
                -> "gpu"
                -> "None" : automatic choice

    _theta_     : float (optional) 
        1D Numpy array of float representing the starting random theta

    Returns
    -------
    out : ndarray
        1D Numpy array of float dtype representing the points on circle

    """

    # check the input
    try:
        # --------------------------------------------------------------------        dataset
        if type(dataset) != np.ndarray:
            raise Circleclustering_input_zero("The array is not an ndarray")
        if len(dataset.shape) == 0 or len(dataset.shape) == 1:
            raise Circleclustering_input_zero("The input shape is incorrect...")
        if dataset.shape[0] <= 1:
            raise Circleclustering_input_zero("The dataset needs two have at least two element")
        # --------------------------------------------------------------------        precision
        if precision == None:
            precision = "low"
        output_str = precision.lower()
        if precision != "high" and precision != "medium" and precision != "low":
            raise Circleclustering_wrong_precision_string("The input string precision is not correct...")
        elif precision == "high":
            eps = 0.0001
        elif precision == "medium":
            eps = 0.001
        elif precision == "low":
            eps = 0.01
        # --------------------------------------------------------------------        hardware

        if hardware == None:
            # automatic search for the hardware
            hardware = "cpu"
        else:
            hardware = hardware.lower()
            if hardware != "cpu" and hardware != "gpu":
                raise Circleclustering_hardware_wrong("the input string hardware is not correct")
        
        
        
        # >--------------------------test passed-------------------------------<
        # >--------------------------------------------------------------------<
        return CircleClustering_test_passed(dataset, target, eps, hardware, _theta_)
        # >--------------------------------------------------------------------<

    except Circleclustering_input_zero as error:
        print('A New Exception occured:', error.value)
        print("Enter an numpy.ndarray with two dimensions")
        return 1
        

    except Circleclustering_wrong_precision_string as error:
        print('A New Exception occured:', error.value)
        print("It represents the Achievable error threshold ")
        print("you can choose between : ")
        print("-> high")
        print("-> medium")
        print("-> low")
        return 2

    except Circleclustering_hardware_wrong as error:
        print('A New Exception occured:', error.value)
        print("Enter one the two possible hardware option : ")
        print("-> cpu")
        print("-> gpu")
        print("leave blank if you want to leave the algorithm the best decision based on your data")
        return 3
    


def CircleClustering_test_passed(dataset, target, eps, hardware, _theta_ = None):
    # get the thetas
    numberOfSamplesInTheDataset = dataset.shape[0]
    if type(_theta_) is not np.ndarray:
        theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
    else:
        theta = _theta_
    
    if hardware == "cpu":
        weights, S, C = cc_cpu.computing_weights(dataset, theta)
        theta = cc_cpu.loop(weights, theta, S, C, eps)
        return weights, S, C, theta, target
    
    if hardware == "gpu":
        weights_gpu, S_gpu, C_gpu = cc_gpu.computing_weights(dataset, theta)
        theta = cc_gpu.loop_gpu(weights_gpu, theta, S_gpu, C_gpu, eps)
        return weights_gpu, S_gpu, C_gpu, theta, target


    