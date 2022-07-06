import numpy as np

def convert_f32(a):
    """
    Convert to float32 dtype.

    Parameters
    ----------
    a : ndarray

    Returns
    -------
    out : ndarray
        Converts to float32 dtype if not already so. This is needed for
        implementations that work exclusively work such datatype.

    """

    if a.dtype!=np.float32:
        return a.astype(np.float32)
    else:
        return a


def getDataFromGpu(weights, S, C):
    return weights.get(), S.get(), C.get()

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def truncatefloat6digits(n):
    txt = f"(n:.6f)"
    y = float(txt)
    return y
 
def trunc(a, x):
    int1 = int(a * (10**x))/(10**x)
    return float(int1)