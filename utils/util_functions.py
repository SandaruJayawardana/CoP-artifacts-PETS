
'''
    Utility functions
'''

import numpy as np

def validate_perturbed_data(original_len, COLUMNS, perturbed_data_list):
    COLUMNS = set(COLUMNS)
    for i in range(len(perturbed_data_list)):
        assert perturbed_data_list[i].shape[0] % original_len == 0, f"Length not match with original: original {original_len}, found {perturbed_data_list[i].shape[0]}."
        assert set(perturbed_data_list[i].columns).issubset(COLUMNS), f"Attributes in i={i} = {set(perturbed_data_list[i].columns)} does not subset of {COLUMNS}."

def _0_1_cal(actual, perturbed):
    N = len(actual)
    loss = 0
    for i_index, i in enumerate(actual):
        if i != perturbed[i_index]:
            loss += 1
    return loss/(N)

def validate_err_matrix(err_matrix, dim):

    if not(isinstance(err_matrix, np.ndarray) and err_matrix.ndim == 2):
        assert False, "Error matrix is not valid numpy array or not a 2 dim matrix"
    elif not(err_matrix.shape[0] == dim and err_matrix.shape[1] == dim):
        assert False, f"Error matrix is not valid. Matrix dim not match. Dim 1 has {err_matrix.shape[0]}, Dim 2 has {err_matrix.shape[1]}, expected {dim}"
 
    if np.all(np.diag(err_matrix) != np.zeros(dim)):
        print("dim ", dim)
        assert False, f"Digonal needs to be zero. But found {np.diag(err_matrix)}"

    # if np.max(err_matrix) > 1:
    #     assert False, f"Matrix should be between 0 and 1. But found {np.max(err_matrix)}"

    if np.min(err_matrix) < 0:
        assert False, f"Matrix should be between 0 and 1. But found {np.min(err_matrix)}"
    
    return True

def validate_distribution(prior_dist, n, terminate = True):
    
    if terminate:
        assert abs(np.sum(prior_dist) - 1) < 0.1, f"Prior distribution should be sum to 1. Found {np.sum(prior_dist):.6f}"
        assert not(np.min(prior_dist) < 0 or np.max(prior_dist) > 1), f"Probabilities should be between 0 and 1. Found {np.min(prior_dist) if np.min(prior_dist) < 0 else np.max(prior_dist)}"
        assert not(len(prior_dist) != n), f"Prior distribution and state count do not match {len(prior_dist)} and {n}"
    
    if not(terminate): 
        return (abs(np.sum(prior_dist) - 1) < 0.1) and (not(np.min(prior_dist) < 0 or np.max(prior_dist) > 1)) and (not(len(prior_dist) != n))
    
    return True

def validate_alphabet(alphabet):
    
    if len(alphabet) != len(set(alphabet)):
        assert False, "Alphabet should contain unique elements"
    return True

def cal_tv(p, q):
    sum_ = np.sum(np.abs(p - q))
    return sum_/2

def divide_range_into_slices(start, end, num_slices):
    slice_size = (end - start) / num_slices
    slices = [start + i*slice_size for i in range(num_slices)]
    slices.append(end)
    return slices

def list_to_string(l):
    s = ""
    for i in l:
        s += str(i) + " "
    return s

def string_to_list(s):
    __l = s.split(" ")
    l = []
    for i in __l:
        if i != "" :
            l.append((i))
    return l
