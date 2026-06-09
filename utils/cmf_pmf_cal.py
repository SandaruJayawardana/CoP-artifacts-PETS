import numpy as np
import pickle

FILE_LOCATION = '/home/sjay9734/CPM-publish/saved_metadata'

def list_to_string(l):
    s = ""
    for i in l:
        s += str(i) + " "
    return s

def get_final_alphabet_list(alphabet_list, base_string = "", output_list = []):
    if len(alphabet_list) == 1:
        for i in alphabet_list[0]:
            if base_string == "":
                output_list.append(str(i))
            else:
                output_list.append(base_string + " " + str(i))
        return output_list
    new_output_list = output_list
    new_base_string = base_string
    for i in alphabet_list[0]:
        if base_string == "":
            new_base_string = str(i)
        else:
            new_base_string = base_string + " " + str(i)
        new_output_list = get_final_alphabet_list(alphabet_list[1:], base_string = new_base_string, output_list = new_output_list)
    return new_output_list

def get_ordered_alphabet_(attributes, original_data):
    alphabet_list = []

    if not(isinstance(attributes, list)):
        attributes = [attributes]

    for attribute in attributes:
        alphabet_list.append(np.unique(original_data[attribute].values))
        # x = np.unique(original_data[attribute].values)
        # try:
        #     x_array = np.array(x, dtype=float)
        #     x_array = np.array(np.sort(x_array), dtype=str)
        # except:
        #     x_array = x
        # alphabet_list.append(x_array)

    return (get_final_alphabet_list(alphabet_list=alphabet_list, base_string = "", output_list = []))

def get_CMF(filtered_original_data, filtered_perturb_data, Z_alphabet, X_k_alphabet, samples):
    probability_matrix = np.ones((len(X_k_alphabet), len(Z_alphabet)))

    index_x_k = 0
    index_z = 0
    len_of_original_dataset = np.shape(filtered_original_data)[0]
    
    for index_i, i in enumerate(filtered_perturb_data[:samples,:]):
        index_z = Z_alphabet.index(list_to_string(i)[:-1])
        index_x_k = X_k_alphabet.index(list_to_string(filtered_original_data[index_i%len_of_original_dataset])[:-1])
        probability_matrix[index_x_k][index_z] += 1
    for i in range(np.shape(probability_matrix)[0]):
        for j in range(np.shape(probability_matrix)[1]):
            if probability_matrix[i][j] < 3:
                probability_matrix[i][j] = 0

    probability_matrix /= np.sum(probability_matrix)
    conditional_matrix = np.transpose(np.transpose(probability_matrix)/np.sum(probability_matrix, axis=1))

    return conditional_matrix, probability_matrix

def validate(CMF_dict, column):
    for index_i, i in enumerate(column[:]):
        for index_j, j in enumerate(column):
            if i == j:
                continue

            key_ = str(i) + ' ' + str(j)
                
            if key_ not in CMF_dict.keys():
                print("Incomplete CMF", key_)
                raise ValueError(f"Error key {index_i}, {index_j}")

def get_cmf_pmf_dict_with_alphabet(data, dataset, name, is_pmf = False, samples=0):
    global FILE_LOCATION
    original_data = data 

    print(original_data.head())
    if samples == 0:
        samples = original_data.shape[0]
    print("CMF Samples ", samples)
    COLUMNS = original_data.columns

    try:
        
        if is_pmf:
            with open(f'{FILE_LOCATION}/{dataset}/{name}_pmf.pkl', 'rb') as file:
                PMF_dict, _, alphabet_dict = pickle.load(file)
                validate(PMF_dict, COLUMNS)  
        else:  
            with open(f'{FILE_LOCATION}/{dataset}/{name}_cmf.pkl', 'rb') as file:
                CMF_dict, _, alphabet_dict = pickle.load(file)
                validate(CMF_dict, COLUMNS)
    except:

        try:
            with open(f'{FILE_LOCATION}/{dataset}/{name}_cmf.pkl', 'rb') as file:
                CMF_dict, C, alphabet_dict  = pickle.load(file)

            with open(f'{FILE_LOCATION}/{dataset}/{name}_pmf.pkl', 'rb') as file:
                PMF_dict, C, alphabet_dict  = pickle.load(file)
        except:
            CMF_dict = {}
            PMF_dict = {}
            alphabet_dict = {}
        
        for index_i, i in enumerate(COLUMNS[:]):
            X_k = [i]
            X_k_alphabet = get_ordered_alphabet_(X_k, original_data=original_data)
            for index_j, j in enumerate(COLUMNS):
                if i == j:
                    continue
                
                Z = [j]
                Z_alphabet = get_ordered_alphabet_(Z, original_data=original_data)

                filtered_original_data = original_data[X_k].values
                filtered_perturb_data = original_data[Z].values

                key_ = str(i) + ' ' + str(j)
                print("key ", index_i, index_j, X_k_alphabet, Z_alphabet)
                if key_ not in CMF_dict.keys():
                    CMF, PMF = get_CMF(filtered_original_data, filtered_perturb_data, Z_alphabet, X_k_alphabet, samples=samples) #original_data.shape[0])
                    CMF_dict[key_] = CMF
                    PMF_dict[key_] = PMF
            alphabet_dict[i] = X_k_alphabet
            with open(f'{FILE_LOCATION}/{dataset}/{name}_cmf.pkl', 'wb') as file:
                pickle.dump((CMF_dict, COLUMNS, alphabet_dict), file)
            with open(f'{FILE_LOCATION}/{dataset}/{name}_pmf.pkl', 'wb') as file:
                pickle.dump((PMF_dict, COLUMNS, alphabet_dict), file)
    if is_pmf:
        return PMF_dict, alphabet_dict
    return CMF_dict, alphabet_dict
