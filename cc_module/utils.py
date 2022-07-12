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
	

def plot_blobs(X, labels=None, threeD=False, doPCA=True, sizex=1):

	#plt.figure(figsize=(10, 10))
	#plt.scatter(X[:, 0], X[:, 1], c=l)
	#plt.title("Blobs")
	#plt.show()

	if threeD:
		if PCA:
			pca = PCA(n_components=3)
			components = pca.fit_transform(X)
		else:
			components = X[:,0:3]  
		if labels is None:
			fig = px.scatter_3d(components, x=0, y=1, z=2, title='Blobs 3D', labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
		else:
			fig = px.scatter_3d(components, x=0, y=1, z=2, color=labels, title='Blobs 3D', labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
	else:
		if doPCA:
			pca = PCA(n_components=2)
			components = pca.fit_transform(X)
		else:
			components = X[:,0:2]  
		if labels is None:
			fig = px.scatter(components, x=0, y=1, title='Blobs 2D', labels={'0': 'PC 1', '1': 'PC 2'})
		else:
			fig = px.scatter(components, x=0, y=1, title='Blobs 2D', color=labels, labels={'0': 'PC 1', '1': 'PC 2'})
	
	fig.update_layout(
		width = 800*sizex,
		height = 800*sizex,
		title = "fixed-ratio axes")
	fig.update_yaxes(
		scaleanchor = "x",
		scaleratio = 1)
	fig.show()  