"""
Show the performance for solving a 2D Poisson problem using CPU Taichi Sparse solver.
"""

import numpy as np
import scipy.io as sio
import time

import taichi as ti

ti.init(arch=ti.cpu)


@ti.kernel
def init_b(b: ti.types.ndarray(), nrows: ti.i32):
    for i in range(nrows):
        b[i] = 1.0 + i / nrows


@ti.kernel
def print_x(x: ti.types.ndarray(), ncols: ti.i32, length: ti.i32):
    for i in range(ncols - length, ncols):
        print(x[i])


print(">> load sparse matrix........")
A_raw_coo = sio.mmread('data/lap2D_5pt_n100.mtx')
nrows, ncols = A_raw_coo.shape
nnz = A_raw_coo.nnz

A_csr = A_raw_coo.tocsr()
b = ti.ndarray(shape=nrows, dtype=ti.f32)
init_b(b, nrows)

print(">> solve Ax = b using CuSparseSolver ......... ")
A_coo = A_csr.tocoo()
triplets = ti.ndarray(shape=nnz * 3, dtype=ti.f32)


@ti.kernel
def init_triplets(triplets: ti.types.ndarray(), row: ti.types.ndarray(),
                  col: ti.types.ndarray(), data: ti.types.ndarray()):
    for i in range(nnz):
        triplets[i * 3 + 0] = row[i]
        triplets[i * 3 + 1] = col[i]
        triplets[i * 3 + 2] = data[i]


init_triplets(triplets, A_coo.row, A_coo.col, A_coo.data)
b_ti = b.to_numpy()
start = time.time()
N = 100
for i in range(N):
    A_ti = ti.linalg.SparseMatrix(n=nrows, m=ncols, dtype=ti.float32)
    A_ti.build_from_ndarray(triplets)
    x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)
    solver = ti.linalg.SparseSolver()
    solver.compute(A_ti)
    x_ti = solver.solve(b_ti)
    ti.sync()
    # print(">> CuSparseSolver results:")
    # print_x(x_ti, ncols, 10)
ti.sync()
end = time.time()
print(">> time:", (end - start) / N)
