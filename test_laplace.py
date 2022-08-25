import taichi as ti
import numpy as np
from scipy.sparse import coo_matrix
import time

ti.init(arch=ti.cpu)

N = 5000

Abuilder = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=150000)


@ti.kernel
def laplace(A: ti.types.sparse_matrix_builder()):
    for i in range(N):
        if i > 0:
            A[i - 1, i] += 1.0
            A[i, i] += 1
        if i < N - 1:
            A[i + 1, i] += 1.0
            A[i, i] += 1.0


# Taichi Sparse Matrix
s = time.time()
laplace(Abuilder)
h = time.time()
A = Abuilder.build()
e = time.time()
print(f"taichi sparse matrix build: {e-s}, builder: {h-s}, build: {e-h}")

# Numpy Dense Matrix
s = time.time()
K = np.zeros(shape=(N, N))
for i in range(N):
    if i > 0:
        K[i - 1, i] += 1.0
        K[i, i] += 1
    if i < N - 1:
        K[i + 1, i] += 1.0
        K[i, i] += 1.0
e = time.time()
print(f"Numpy Dense Matrix build: {e - s}")

# Scipy Sparse Matrix
s = time.time()
row = np.zeros(3 * N)
col = np.zeros(3 * N)
data = np.zeros(3 * N)
for i in range(N):
    row[3 * i:3 * i + 3] = [i, i, i]
    col[3 * i:3 * i + 3] = [i - 1, i, i + 1]
    data[3 * i:3 * i + 3] = [1, 2, 1]

A = coo_matrix((data[1:N - 1], (row[1:N - 1], col[1:N - 1])), shape=(N, N))
e = time.time()
print(f"Scipy Sparse Matrix build: {e - s}")
