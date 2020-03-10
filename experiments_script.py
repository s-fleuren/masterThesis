from main_methods import *
import os
import time
from tabulate import tabulate

np.random.seed(1901714035)

directory_name = 'datasets/MPEG7_CE-Shape-1_Part_B/MPEG7_CE-Shape-1_Part_B'
directory = os.fsencode(directory_name)
image_names = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".gif"):
        im = Image.open(directory_name + '/' + filename)
        imarray = np.array(im)
        n1, n2 = imarray.shape
        N = n1 * n2
        if N <= 4000:
            image_names.append(filename[:-4])

image_names.sort()
image_properties = []
for image_name in image_names:
    filename = image_name + '.gif'
    im = Image.open(directory_name + '/' + filename)
    imarray = np.array(im)
    n1, n2 = imarray.shape
    N = n1 * n2
    white_pixel_counter = 0
    for i in range(n1):
        for j in range(n2):
            if imarray[i, j] > 0:
                white_pixel_counter += 1
    image_properties.append([image_name, n1, n2, N, white_pixel_counter/N])

print(tabulate(image_properties, tablefmt="latex_raw", headers=['Image name', '$n_1$', '$n_2$', '$N$', 'density'],
               floatfmt=".4f"))


output_directory_name = 'D:/documenten/scriptie/experiments_output/'
result_table = []
timings_table = []
x_properties_table = []

tol = 1e-6
rounding_parameter = 2
for image_name in image_names:
    filename = image_name + '.gif'
    im = Image.open(directory_name + '/' + filename)
    imarray = np.array(im)
    n1, n2 = imarray.shape
    N = n1 * n2
    true_x = imarray.reshape(N)

    for i in range(N):
        if true_x[i] == 255:
            true_x[i] = 1


    for m in range(2, 5):
        x_properties_row = [image_name, m, int(0)]
        used_projections = [1] * m + [0] * (4 - m)
        A = combineProjections(n1, n2, used_projections)
        M = A.shape[0]
        y = A @ true_x
        ccls = CCLS_instance(A, y, m)
        ccls.reduction_size_one_autarkies()
        current_N = ccls.A.shape[1]
        x_0 = 0.5 * np.ones(current_N)
        x_hat = compute_constrained_least_squares_solution(ccls.A, ccls.y, x_0, tol).x
        x_hat_full = np.insert(x_hat, ccls.mask, ccls.z)
        im_x_hat = Image.fromarray(255 * x_hat_full.reshape((n1, n2)))
        im_x_hat.convert('RGB').save(output_directory_name + image_name + '_' + str(m) + '_' + str(0.00) +
                                     '_x_hat' + '.png')
        x_tilde = rounding(x_hat, ccls.A, ccls.y)
        x_tilde_full = np.insert(x_tilde, ccls.mask, ccls.z)
        im_x_tilde = Image.fromarray(255 * x_tilde_full.reshape((n1, n2)))
        im_x_tilde.convert('RGB').save(output_directory_name + image_name + '_' + str(m) + '_' + str(0.00) +
                                       '_x_tilde' + '.png')
        x_properties_row.append(0.5 * np.linalg.norm(A @ x_hat_full - y) ** 2)
        x_properties_row.append(0.5 * np.linalg.norm(A @ x_tilde_full - y) ** 2)
        x_properties_row.append(compare_vectors(x_tilde_full, true_x))

        x_properties_table.append(x_properties_row)

        for noise_level in range(1, 4):
            sigma = 0.01 * (2 ** noise_level)
            x_properties_row = [image_name, m, sigma]
            table_row = [image_name, m, sigma]
            table_row_timings = [image_name, m, sigma]
            e = np.random.normal(1, sigma, M)
            y = np.around(e * (A @ true_x), rounding_parameter)
            ccls = CCLS_instance(A, y, m)
            time_0 = time.process_time()
            ccls.reduction_size_one_autarkies()
            time_1 = time.process_time()
            ccls.save_z_image(n1, n2, output_directory_name + image_name + '_' + str(m) + '_' + str(sigma) + '_a' + '.png')
            table_row.append(ccls.z.size)
            table_row.append(ccls.compare_z(true_x))
            table_row_timings.append(time_1 - time_0)

            ccls = CCLS_instance(A, y, m)
            time_0 = time.process_time()
            ccls.reduction_size_one_autarkies()
            current_N = ccls.A.shape[1]
            x_0 = 0.5 * np.ones(current_N)
            x_hat = compute_constrained_least_squares_solution(ccls.A, ccls.y, x_0, tol).x
            x_hat_full = np.insert(x_hat, ccls.mask, ccls.z)
            ccls.reduction_dual(tol, 0, precomputed_x_hat=x_hat)
            time_1 = time.process_time()
            im_x_hat = Image.fromarray(255 * x_hat_full.reshape((n1, n2)))
            im_x_hat.convert('RGB').save(output_directory_name + image_name + '_' + str(m) + '_' + str(sigma) +
                                         '_x_hat' + '.png')
            ccls.save_z_image(n1, n2, output_directory_name +
                              image_name + '_' + str(m) + '_' + str(sigma) + '_b' + '.png')
            table_row.append(ccls.z.size)
            table_row.append(ccls.compare_z(true_x))
            table_row_timings.append(time_1 - time_0)

            ccls = CCLS_instance(A, y, m)
            time_0 = time.process_time()
            ccls.reduction_size_one_autarkies()
            current_N = ccls.A.shape[1]
            x_0 = 0.5 * np.ones(current_N)
            x_hat = compute_constrained_least_squares_solution(ccls.A, ccls.y, x_0, tol).x
            x_hat_full = np.insert(x_hat, ccls.mask, ccls.z)
            x_tilde = rounding(x_hat, ccls.A, ccls.y)
            x_tilde_full = np.insert(x_tilde, ccls.mask, ccls.z)
            im_x_tilde = Image.fromarray(255 * x_tilde_full.reshape((n1, n2)))
            im_x_tilde.convert('RGB').save(output_directory_name + image_name + '_' + str(m) + '_' + str(sigma) +
                                         '_x_tilde' + '.png')
            upperbound = 0.5 * np.linalg.norm(ccls.A @ x_tilde - ccls.y) ** 2
            ccls.reduction_dual(tol, upperbound, precomputed_x_hat=x_hat)
            time_1 = time.process_time()
            ccls.save_z_image(n1, n2,
                              output_directory_name + image_name + '_' + str(m) + '_' + str(sigma) + '_c' + '.png')
            table_row.append(ccls.z.size)
            table_row.append(ccls.compare_z(true_x))
            table_row_timings.append(time_1 - time_0)

            ccls = CCLS_instance(A, y, m)
            time_0 = time.process_time()
            ccls.reduction_size_one_autarkies()
            ccls.reduction_implication_network(rounding_parameter)
            time_1 = time.process_time()
            ccls.save_z_image(n1, n2,
                              output_directory_name + image_name + '_' + str(m) + '_' + str(sigma) + '_d' + '.png')
            table_row.append(ccls.z.size)
            table_row.append(ccls.compare_z(true_x))
            table_row_timings.append(time_1 - time_0)
            x_properties_row.append(0.5 * np.linalg.norm(A @ x_hat_full - y) ** 2)
            x_properties_row.append(0.5 * np.linalg.norm(A @ x_tilde_full - y) ** 2)
            x_properties_row.append(compare_vectors(x_tilde_full, true_x))

            x_properties_table.append(x_properties_row)
            result_table.append(table_row)
            timings_table.append(table_row_timings)

print(tabulate(result_table, tablefmt="latex_raw",
               headers=['Image name', '$m$', '$\sigma$', '$\\abs{I_z}$', '$a(z, \chi)$',
                        '$\\abs{I_z}$', '$a(z, \chi)$', '$\\abs{I_z}$', '$a(z, \chi)$',
                        '$\\abs{I_z}$', '$a(z, \chi)$'], floatfmt=".2f"))
print(tabulate(timings_table, tablefmt="latex_raw",
               headers=['Image name', '$m$', '$\sigma$', 'time (s)', 'time (s)', 'time (s)', 'time (s)']))
print(tabulate(x_properties_table, tablefmt="latex_raw",
               headers=['Image name', '$m$', '$\sigma$', '$f_{LS}(\hat x)$', '$f_{LS}(\Tilde x)$', '$a(\Tilde x, \chi)$'],
               floatfmt=".2f"))