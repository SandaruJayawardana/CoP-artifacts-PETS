import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.mutual_information import mutualinformationempirical, entropyempirical

FILE_LOCATION = '/home/sjay9734/CPM-publish/saved_metadata'

def get_mi_dict(dataset, COLUMNS, data, name = ""):

    global FILE_LOCATION

    try:
        with open(f'{FILE_LOCATION}/{dataset}/{dataset}_{name}_mi.pkl', 'rb') as file:
            mi_dict, COLUMNS  = pickle.load(file)

    except:
        data = data.copy()
        for attr in COLUMNS:
            le = LabelEncoder()
            data[attr] = le.fit_transform(data[attr])

        mi_dict = {}

        for index_i, i in enumerate(COLUMNS):
            for index_j, j in enumerate(COLUMNS):
                if i == j:
                    continue
                key_ = i + " " + j
                key_2 = j + " " + i

                if key_ in mi_dict.keys():
                    continue

                entropy = entropyempirical(np.array(data[[i, j]].values, dtype=np.float64))[0]
                normalised_mi = mutualinformationempirical(np.array(data[i].values, dtype=np.float64), np.array(data[j].values, dtype=np.float64))[0]/entropy
                # print("H, I", entropy, mutualinformationempirical(np.array(data[i].values, dtype=np.float64), np.array(data[j].values, dtype=np.float64))[0], normalised_mi)
                mi_dict[key_] = normalised_mi
                mi_dict[key_2] = normalised_mi
        
        with open(f'{FILE_LOCATION}/{dataset}/{dataset}_{name}_mi.pkl', 'wb') as file:
            pickle.dump((mi_dict, COLUMNS), file)
    return mi_dict
