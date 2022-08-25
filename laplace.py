import taichi as ti

ti.init(arch=ti.cpu)

N = 16

x, y = ti.field(ti.f32), ti.field(ti.f32)

ti.root.dense(ti.ij, N).place(x, y)


@ti.kernel
def laplace():
    for i, j in x:
        if (i + j) % 3 == 0 and 1 <= i < N - 1 and 1 <= j < N - 1:
            y[i, j] = 4.0 * x[i, j] \
                    - x[i - 1, j] - x[i + 1, j] - x[i, j - 1] - x[i, j + 1]
        else:
            y[i, j] = 0.0


for i in range(N):
    for j in range(N):
        x[i, j] = i + j

laplace()

for i in range(N):
    for j in range(N):
        print(y[i, j], sep=" ")
    print()
