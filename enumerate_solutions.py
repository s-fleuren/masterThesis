import numpy as np
import pickle
from projection_matrices import *
from utility import *


def get_unique_and_muliples_N(d):
    n_u = 0
    n_m = 0
    for k in d.keys():
        if len(d[k]) == 0:
            print('error')
        elif len(d[k]) == 1:
            n_u += 1
        else:
            n_m += len(d[k])
    return [n_u, n_m]


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


for m in range(2, 5):
    for n in range(2, 5):
        used_projections = [1] * m + [0] * (4 - m)
        A = combineProjections(n, n, used_projections)
        print(A.todense())
        M = A.shape[0]
        N = n**2
        T = {}
        for i in range(2 ** N):
            x = unpackbits16(i, N)
            y = A * x
            T.setdefault(tuple(y), [])
            T[tuple(y)].append(i)
        save_obj(T, 'solutions_dict_m' + str(m) + 'n' + str(n))
        print(get_unique_and_muliples_N(T))


