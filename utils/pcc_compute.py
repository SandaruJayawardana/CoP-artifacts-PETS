import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.mutual_information import mutualinformationempirical, entropyempirical

FILE_LOCATION = '/home/sjay9734/CPM-publish/saved_metadata'

def pcc_from_joint(joint):
    n, axes = joint.ndim, range(joint.ndim)
    means  = [ (np.arange(k) * joint.sum(axis=tuple(a for a in axes if a != i))).sum()
               for i, k in enumerate(joint.shape) ]
    vars_  = [ (np.arange(k)**2 * joint.sum(axis=tuple(a for a in axes if a != i))).sum() - μ**2
               for (i, k), μ in zip(enumerate(joint.shape), means) ]

    corr = np.eye(n)
    for i in axes:
        for j in range(i+1, n):
            pair = joint.sum(axis=tuple(a for a in axes if a not in (i, j)))
            xi, xj = np.meshgrid(np.arange(joint.shape[i]), np.arange(joint.shape[j]), indexing="ij")
            cov = (xi * xj * pair).sum() - means[i] * means[j]
            corr[i, j] = corr[j, i] = cov / np.sqrt(vars_[i] * vars_[j])
    return corr

def get_pcc_dict(dataset, COLUMNS, joint_dict, name = ""):

    global FILE_LOCATION

    try:
        with open(f'{FILE_LOCATION}/{dataset}/{dataset}_{name}_pcc.pkl', 'rb') as file:
            pcc_dict, COLUMNS  = pickle.load(file)

    except:
        pcc_dict = {}

        for index_i, i in enumerate(COLUMNS):
            for index_j, j in enumerate(COLUMNS):
                if i == j:
                    continue
                key_ = i + " " + j
                key_2 = j + " " + i

                if key_ in pcc_dict.keys():
                    continue
                pcc = abs(pcc_from_joint(joint_dict[key_]))[1,0]
                pcc_dict[key_] = pcc
                pcc_dict[key_2] = pcc
        
        with open(f'{FILE_LOCATION}/{dataset}/{dataset}_{name}_pcc.pkl', 'wb') as file:
            pickle.dump((pcc_dict, COLUMNS), file)
    return pcc_dict
