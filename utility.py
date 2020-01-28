import numpy as np


def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p. Taken from
    https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s