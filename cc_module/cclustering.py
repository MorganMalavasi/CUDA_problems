import numpy as np
import cclustering_cpu as cc_cpu
import cclustering_gpu as gg_cpu

# constants
PI = np.pi

class Circleclustering_input_zero:
    """Raised when the input dataset shapes are zero"""
    pass

class Circleclustering_wrong_precision_string:
    """Raised when the input precision string is wrong"""
    pass

class Circleclustering_harware_wrong:
    """Raised when the input type harware string is wrong"""
    pass
    

def CircleClustering(dataset, target = None, precision = None, hardware = None):
    
    """
    Compute the points on the circle representing the dataset

    Parameters
    ----------
    
    dataset : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
        The number of columns is the number of features (dimension)
    
    target  : ndarray
        1D Numpy array of int dtype representing the class of each point in
        the dataset
    
    precision     : string
        Achievable error threshold.
        3 types ->  "high"    (0.0001)
                ->  "medium"  (0.001)
                ->  "low"     (0.01)

    Returns
    -------
    out : ndarray
        1D Numpy array of float dtype representing the points on circle

    """

    # check the input
    try:
        if dataset.shape[0] == 0 or dataset.shape[1] == 0:
            raise Circleclustering_input_zero
        # --------------------------------------------------------------------        

        output_str = precision.lower()
        if precision != "high" and precision != "medium" and precision != "low":
            raise Circleclustering_wrong_precision_string
        # --------------------------------------------------------------------

        if hardware == None:
            hardware = chooseHardware()
        else:
            hardware = hardware.lower()
            if hardware != "cpu" and hardware != "gpu":
                raise Circleclustering_harware_wrong
        # --------------------------------------------------------------------
        
        
        # >--------------------------test passed-------------------------------<
        # >--------------------------------------------------------------------<
        CircleClustering_test_passed(dataset, target, eps, hardware)
        # >--------------------------------------------------------------------<

    except Circleclustering_input_zero:
        print("The input shape is incorrect...")
        print("dataset need to be of size > 0 with number of features > 0")

    except Circleclustering_wrong_string:
        print("The input string precision is not correct...")
        print("It represents the Achievable error threshold ")
        print("you can choose between : ")
        print("-> high")
        print("-> medium")
        print("-> low")

    except Circleclustering_harware_wrong:
        print("Enter one the two possible hardware option : ")
        print("-> cpu")
        print("-> gpu")
        print("leave blanck if you want to leave the algorithm the best decision based on your data")
    


def CircleClustering_test_passed(dataset, target, eps, hardware):
    # get the thetas
    numberOfSamplesInTheDataset = dataset.shape[0]
    theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)

    if hardware == "cpu":
        weights = cc_cpu.computing_weights(dataset)
        S, C = cc_cpu.C_S(weights, theta)
        theta = cc_cpu.loop(weights, theta, S, C, eps)
        return theta, target

    if hardware == "gpu":
        weights = cc_gpu.computing_weights(dataset)
        S, C = cc_gpu.C_S(weights, theta)
        theta = cc_gpu.loop_jit(weights, theta, S, C, eps)
        return theta, target

    
        
    

