from PIL import Image
from projection_matrices import *
from scipy_least_squares import *
from utility import *

im = Image.open('datasets/bg1-bg10/bg9.tif')
imarray = np.array(im)
n1, n2 = imarray.shape
N = n1 * n2
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

x_0 = 0.5 * np.ones(N)
tol = 1e-4
m = 4
used_projections = [1] * m + [0] * (4 - m)
A = combineProjections(n1, n2, used_projections)
y = A @ x
result = compute_constrained_least_squares_solution(A, y, x_0, tol, verbose=2)
im2 = Image.fromarray(255 * result.x.reshape((n1, n2)))
im2.show()
rounded_x = rounding(result.x, A, y)
im3 = Image.fromarray(255 * rounded_x.reshape((n1, n2)))
im3.show()

print(result.cost)
print(np.linalg.norm(A @ result.x - y)**2)
print(np.linalg.norm(A @ rounded_x - y)**2)
