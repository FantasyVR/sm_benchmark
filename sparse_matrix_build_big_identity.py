import taichi as ti
import numpy as np
from scipy.sparse import coo_matrix
import time

ti.init(arch=ti.cpu)

N = 10000

Abuilder = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=100000000)


@ti.kernel
def identity(A: ti.types.sparse_matrix_builder()):
    for i in range(N):
        for j in range(N):
            A[i, j] += 1.0


# Taichi Sparse Matrix
identity(Abuilder)
h = time.time()
A = Abuilder.build()
s = time.time()
identity(Abuilder)
h = time.time()
A = Abuilder.build()
e = time.time()
print(f"taichi sparse matrix build: {e-s}, builder: {h-s}, build: {e-h}")

# Numpy Dense Matrix
s = time.time()
K = np.ones(shape=(N, N))
e = time.time()
print(f"Numpy Dense Matrix build: {e - s}")

# Scipy Sparse Matrix
s = time.time()
row = np.zeros(N * N)
col = np.zeros(N * N)
data = np.ones(N * N)
for i in range(N):
    for j in range(N):
        row[i * N + j] = i
        col[i * N + j] = j

A = coo_matrix((data, (row, col)), shape=(N, N))
e = time.time()
print(f"Scipy Sparse Matrix build: {e - s}")
