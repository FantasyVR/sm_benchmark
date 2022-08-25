"""
  We compare the performance of Scipy sparse solver and taichi sparse solver when solving a 13514 x 13514 3D Poisson problem.
"""
import taichi as ti
import numpy as np
import scipy as sp
import scipy.io
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time

ti.init(arch=ti.cpu)

# Download from https://sparse.tamu.edu/FEMLAB/poisson3Da
poissonProblem = scipy.io.loadmat('data/poisson3Db.mat')['Problem']

name = poissonProblem['name'][0][0][0]
title = poissonProblem['title'][0][0][0]
kind = poissonProblem['kind'][0][0][0]

A = poissonProblem['A'][0][0]
b = poissonProblem['b'][0][0]
N = b.shape[0]  # dimension

print(
    f"Problem Kind: {kind} \n{title} \n >> A: {A.shape}, b: {b.shape} \n >> Number of Non-Zeros: {A.nnz}"
)

s = time.time()
x = spsolve(A, b)
e = time.time()
print(f"Scipy solve time:  {e-s}")

Atriplets = A.tocoo()
nnz = Atriplets.nnz
row = Atriplets.row
col = Atriplets.col
data = Atriplets.data
Abuilder = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=400000)


@ti.kernel
def buildA(Abuilder: ti.types.sparse_matrix_builder(), row: ti.types.ndarray(),
           col: ti.types.ndarray(), data: ti.types.ndarray()):
    for i in range(nnz):
        Abuilder[row[i], col[i]] += data[i]


buildA(Abuilder, row, col, data)

A_taichi = Abuilder.build()

solver = ti.linalg.SparseSolver(solver_type="LDLT")

solver.analyze_pattern(A_taichi)
solver.factorize(A_taichi)
x = solver.solve(b)
s = time.time()
solver.analyze_pattern(A_taichi)
ap = time.time()
solver.factorize(A_taichi)
fac = time.time()
x = solver.solve(b)
e = time.time()
isSuccess = solver.info()
if isSuccess:
    print(
        f"taichi solve time: {e - s}, analyze_pattern: {ap - s}, factorize: {fac- s}, solve: {e - fac}"
    )
