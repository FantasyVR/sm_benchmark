import cupy as cp
import time
# Numpy Dense Matrix
N = 5000

s = time.time()
K = cp.zeros(shape=(N, N))
for i in range(N):
    if i > 0:
        K[i - 1, i] += 1.0
        K[i, i] += 1
    if i < N - 1:
        K[i + 1, i] += 1.0
        K[i, i] += 1.0
e = time.time()
print(f"Cupy Dense Matrix build: {e - s}")
