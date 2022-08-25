import taichi as ti
import numpy as np
import scipy as sp
import scipy.sparse
import time

ti.init(arch=ti.x64)

n = 10000


@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 1.0


# Taichi Sparse Matrix
s = time.time()
K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n)
A = ti.linalg.SparseMatrix(n, n)
fill(K)
K.build(A)
e = time.time()
print(f"taichi sparse matrix build: {e-s}")

# Numpy Dense Matrix
s = time.time()
K = np.diag(np.full(n, 1.0))
e = time.time()
print(f"Numpy Dense Matrix build: {e - s}")

# Scipy Sparse Matrix
s = time.time()
K = sp.sparse.spdiags(np.full(n, 1.0), 0, n, n)
e = time.time()
print(f"Scipy Sparse Matrix build: {e - s}")

#Cupy and GPU
# s = time.time()
# K = cp.eye(2,2)
# e = time.time()
# print(f"Cupy Dense Matrix build: {e - s}")
