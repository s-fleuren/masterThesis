from PIL import Image
from projection_matrices import *
from gradient_projection_method import *
from utility import *

im = Image.open('datasets/bg1-bg10/bg9.tif')
imarray = np.array(im)
n1, n2 = imarray.shape
N = n1 * n2
print(N)
for i in range(n1):
    for j in range(n2):
        if imarray[i, j] <= 90:
            imarray[i, j] = 0
        else:
            imarray[i, j] = 255

im = Image.fromarray(imarray)
im.show()

x = imarray.reshape(N)
for i in range(N):
    if x[i] == 255:
        x[i] = 1

x_0 = Iterate(0.5 * np.ones(N))
tol = 1e-9
m = 4
used_projections = [1] * m + [0] * (4 - m)
A = combineProjections(n1, n2, used_projections)
y = A @ x
cost = 0.5 * np.linalg.norm(A @ x_0.solution - y) ** 2
print(cost)
while cost > 1:
    x_0.gradientProjectionStep(A, y)
    cost = 0.5 * np.linalg.norm(A @ x_0.solution - y) ** 2
    print(cost)
im2 = Image.fromarray(255 * x_0.solution.reshape((n1, n2)))
im2.show()
rounded_x = np.zeros_like(x_0.solution)
for i in range(N):
    if x_0.solution[i] >= 0.5:
        rounded_x[i] = 1
im3 = Image.fromarray(255 * rounded_x.reshape((n1, n2)))
im3.show()

print("\n")
print(0.5 * np.linalg.norm(A @ rounded_x - y) ** 2)
