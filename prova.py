import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2, 3], [4, 5, 6]])

print(A)
print(np.einsum('ij, ij->i', A, A))

dist = np.einsum('ij,ij->i',A,A)[:,np.newaxis] \
     + np.einsum('ij,ij->i',B,B) \
     - 2*np.dot(A,B.T)

print (dist)