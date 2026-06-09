import numpy as np

def error_cal(actual, perturbed, alphabet_dict={}, err_type = "0_1", check = False):
    if check and isinstance(actual, str):
        actual = list(alphabet_dict.keys()).index(actual)
        perturbed = list(alphabet_dict.keys()).index(perturbed)
    # print(actual, perturbed)
    if err_type == "0_1":
        # printactual, perturbed)
        return 0 if (str(actual) == str(perturbed)) else 1
    elif err_type == "l1":
        return abs(actual - perturbed) # np.linalg.norm((np.array(actual)-np.array(perturbed)), 1) # abs((actual)-(perturbed)) # np.linalg.norm(((actual)-(perturbed)), 1)
    elif err_type == "l2":
        return ((actual)-(perturbed))**2
        # return np.linalg.norm((np.array(actual)-np.array(perturbed)), 2) # ((actual)-(perturbed))**2 # np.linalg.norm(((actual)-(perturbed)), 2)
    elif err_type == "mse":
        return np.linalg.norm(((actual)-(perturbed)), 2) # ((actual)-(perturbed))**2 # np.linalg.norm(((actual)-(perturbed)), 2)
    else:
        raise TypeError(f"Unknown error type {err_type}")

class Normalize_error_matrix:
    def __init__(self, attribute_list, alphabet, priority_dict, alphabet_dict = {}, err_type = "0_1", skip_error_matrix = False):
        # print(alphabet)
        self.attribute_list = attribute_list
        self.alphabet = alphabet
        self.priority_dict = priority_dict
        self.alphabet_dict = alphabet_dict
        self.err_type = err_type

        self.normalized_error_matrix = np.zeros((len(alphabet), len(alphabet)))
        # print(np.shape(self.normalized_error_matrix))
        self.__max_value = 1
        self.skip_error_matrix = skip_error_matrix
        if not(skip_error_matrix):
            for index_i, i in enumerate(alphabet): # input
                for index_j, j in enumerate(alphabet):
                    err = self.err(index_i, index_j) #(i, j)
                    self.normalized_error_matrix[index_i][index_j] = err
            self.__max_value = np.max(self.normalized_error_matrix)
            self.normalized_error_matrix /= 1 #np.max(self.normalized_error_matrix, axis=0) # self.__max_value #self.__max_value

    # Original
    def err(self, actual, perturbed):
        # print("actual", actual, "perturbed", perturbed)

        if isinstance(actual, str):
            actual_split = actual.split(" ")
            actual = []
            # print(actual_split)
            for i in actual_split:
                # print(i)
                if i != "" :
                    actual.append((i))
            actual = np.array(actual)

            perturbed_split = perturbed.split(" ")
            perturbed = []
            for i in perturbed_split:
                if i != "":
                    perturbed.append((i))
            perturbed = np.array(perturbed)
        
        if isinstance(actual, np.int64):
            actual = [str(actual)]
            perturbed = [str(perturbed)]

        err = 0
        # print("after actual", actual, "perturbed", perturbed)
        if isinstance(actual, int):
            return error_cal(actual=actual, perturbed=perturbed, err_type=self.err_type)
        for k in range(len(actual)):
            attribute = self.attribute_list[k]
            # print("attribute", attribute)
            if attribute in list(self.priority_dict.keys()):
                priority = self.priority_dict[attribute]
            else:
                priority = 1
            err += error_cal(actual=actual[k], perturbed=perturbed[k], err_type=self.err_type)*priority
        return err
    
    def get_value_error(self, actual, perturbed):
        # print("self.alphabet ", self.alphabet)
        # print("actual ", actual, " perturbed ", perturbed)
        if self.skip_error_matrix:
            return self.err(actual, perturbed)
        
        actual = self.alphabet.index(actual)
        perturbed = self.alphabet.index(perturbed)
        
        return self.normalized_error_matrix[actual][perturbed]
        