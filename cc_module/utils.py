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