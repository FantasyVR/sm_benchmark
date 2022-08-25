import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.time import repeat
import scipy.io
import time

# Download from https://sparse.tamu.edu/FEMLAB/poisson3Da
poissonProblem = scipy.io.loadmat('poisson3Da.mat')['Problem']

name = poissonProblem['name'][0][0][0]
title = poissonProblem['title'][0][0][0]
kind = poissonProblem['kind'][0][0][0]

A = poissonProblem['A'][0][0]
b = poissonProblem['b'][0][0]
N = b.shape[0]  # dimension

print(
    f"Problem Kind: {kind} \n{title} \n >> A: {A.shape}, b: {b.shape} \n >> Number of Non-Zeros: {A.nnz}"
)

A_gpu = csr_matrix(A)
b_gpu = cp.array(b, dtype=float, copy=True)


def compute(A_gpu, b_gpu):
    x = spsolve(A_gpu, b_gpu)


print(repeat(compute, (
    A_gpu,
    b_gpu,
), n_repeat=20))
