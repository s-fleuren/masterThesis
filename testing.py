from main_methods import *

im = Image.open('datasets/MPEG7_CE-Shape-1_Part_B/MPEG7_CE-Shape-1_Part_B/bell-2.gif')
imarray = np.array(im)
n1, n2 = imarray.shape
N = n1 * n2

#for i in range(n1):
#    for j in range(n2):
#        if imarray[i, j] <= 90:
#            imarray[i, j] = 0
#        else:
#            imarray[i, j] = 255

im = Image.fromarray(imarray)
im.show()

x = imarray.reshape(N)
#x = np.array([1, 1, 0, 0])

for i in range(N):
    if x[i] == 255:
        x[i] = 1

m = 4
used_projections = [1] * m + [0] * (4 - m)
A = combineProjections(n1, n2, used_projections)
y = A @ x

ccls = CCLS_instance(A, y, m)
ccls.reduction_size_one_autarkies()
ccls.visualise_z(n1, n2)
ccls.reduction_implication_network()
ccls.visualise_z(n1, n2)
