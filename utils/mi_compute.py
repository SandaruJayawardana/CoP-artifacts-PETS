import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

CURRENT_DIR = Path.cwd()

if CURRENT_DIR.name == "notebooks":
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR

FILE_LOCATION = f'{PROJECT_ROOT}/saved_metadata'

from utils.mutual_information import mutualinformationempirical, entropyempirical

FILE_LOCATION = f'{PROJECT_ROOT}/saved_metadata'

def get_mi_dict(COLUMNS, data):

    global FILE_LOCATION

    try:
        with open(f'{FILE_LOCATION}/dummy.pkl', 'rb') as file:
            mi_dict, COLUMNS  = pickle.load(file)

    except:
        data = data.copy()
        for attr in COLUMNS:
            le = LabelEncoder()
            data[attr] = le.fit_transform(data[attr])

        mi_dict = {}

        for i in COLUMNS:
            for j in COLUMNS:
                if i == j:
                    continue
                key_ = i + " " + j
                key_2 = j + " " + i

                if key_ in mi_dict.keys():
                    continue

                entropy = entropyempirical(np.array(data[[i, j]].values, dtype=np.float64))[0]
                normalised_mi = mutualinformationempirical(np.array(data[i].values, dtype=np.float64), np.array(data[j].values, dtype=np.float64))[0]/entropy
                mi_dict[key_] = normalised_mi
                mi_dict[key_2] = normalised_mi
        
        with open(f'{FILE_LOCATION}/dummy.pkl', 'wb') as file:
            pickle.dump((mi_dict, COLUMNS), file)
    return mi_dict
