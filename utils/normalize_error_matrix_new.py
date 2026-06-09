import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def error_cal(actual, perturbed, err_type = "0_1"):

    if err_type == "0_1":
        return 0 if (str(actual) == str(perturbed)) else 1
    elif err_type == "l2":
        return ((actual)-(perturbed))**2
    else:
        raise TypeError(f"Unknown error type {err_type}")


class Normalize_error_matrix:
    def __init__(self, alphabet, err_type = "0_1"):

        self.alphabet = alphabet
        self.err_type = err_type

        self.normalized_error_matrix = np.zeros((len(alphabet), len(alphabet)))

        self.__max_value = 1

        for index_i, i in enumerate(alphabet): # input
            for index_j, j in enumerate(alphabet):

                if err_type == "0_1":
                    err = error_cal(actual=index_i, perturbed=index_j, err_type=self.err_type) # self.err(index_i, index_j) 
                else:
                    print("alphabet err mat ", alphabet)
                    err = error_cal(actual=float(i), perturbed=float(j), err_type=self.err_type) # self.err(i, j) 
                self.normalized_error_matrix[index_i][index_j] = err

        self.__max_value = np.max(self.normalized_error_matrix)
        self.normalized_error_matrix /= self.__max_value

        sns.heatmap(self.normalized_error_matrix)
        plt.show()
