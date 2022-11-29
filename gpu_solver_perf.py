import numpy as np
import scipy.io as sio

import taichi as ti
import time
ti.init(arch=ti.cuda)


@ti.kernel
def init_b(b: ti.types.ndarray(), nrows: ti.i32):
    for i in range(nrows):
        b[i] = 1.0 + i / nrows


@ti.kernel
def print_x(x: ti.types.ndarray(), ncols: ti.i32, length: ti.i32):
    for i in range(ncols - length, ncols):
        print(x[i])


print(">> load sparse matrix........")
A_raw_coo = sio.mmread('misc/lap2D_5pt_n100.mtx')
nrows, ncols = A_raw_coo.shape
nnz = A_raw_coo.nnz

A_csr = A_raw_coo.tocsr()
b = ti.ndarray(shape=nrows, dtype=ti.f32)
init_b(b, nrows)

A_coo = A_csr.tocoo()
d_row_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_col_coo = ti.ndarray(shape=nnz, dtype=ti.i32)
d_val_coo = ti.ndarray(shape=nnz, dtype=ti.f32)
d_row_coo.from_numpy(A_coo.row)
d_col_coo.from_numpy(A_coo.col)
d_val_coo.from_numpy(A_coo.data)


A_builder = ti.linalg.SparseMatrixBuilder(nrows, ncols, nnz, ti.f32)

@ti.kernel 
def fill(A:ti.linalg.sparse_matrix_builder(), d_row_coo: ti.types.ndarray(), d_col_coo: ti.types.ndarray(), d_val_coo: ti.types.ndarray(), nnz: ti.i32):
    for i in range(nnz):
        row, col = d_row_coo[i], d_col_coo[i]
        A[row, col] += d_val_coo[i]

fill(A_builder, d_row_coo, d_col_coo, d_val_coo, nnz)
print(">> fill sparse matrix........")
A_ti = A_builder.build()
print(">> end filling sparse matrix........")
x_ti = ti.ndarray(shape=ncols, dtype=ti.float32)
solver = ti.linalg.SparseSolver()
print(">> solve Ax = b using CuSparseSolver ......... ")
start = time.time()
for i in range(20):
    solver.analyze_pattern(A_ti)
    solver.factorize(A_ti)
    x_ti = solver.solve(b)
    ti.sync()
end = time.time()
print("avg time: ", (end - start) / 20)


print(">> CuSparseSolver results:")
print_x(x_ti, ncols, 10)

