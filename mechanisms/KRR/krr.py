import numpy as np
from mechanisms.privacy_mechanism import *
from utils.util_functions import *

def random_response_mechanism(alphabet_size, eps):
    vector = (np.exp(eps))*np.ones(alphabet_size)-1
    q =  (np.diag(vector) + 1)/(alphabet_size - 1 + np.exp(eps))
    return q

class Randomized_Response(Privacy_Mechanism):
    
    def __init__(self, STATE_COUNT, INPUT_ALPHABET = [], normalized_objective_err_matrix = None, cache_on = False):
        self.STATE_COUNT = STATE_COUNT

        if len(INPUT_ALPHABET) != STATE_COUNT:
            self.INPUT_ALPHABET = str(range(STATE_COUNT))
        elif validate_alphabet(INPUT_ALPHABET):
            self.INPUT_ALPHABET = INPUT_ALPHABET
        self.cache_on = cache_on
        self.__mechanism_dict = {}
        self.__mechanism = 0
        self.__eps = -1
        self.normalized_objective_err_matrix = normalized_objective_err_matrix
        self.prior_dist = None
    
    def get_mechanism(self, eps):
        if self.cache_on:
            try:
                return self.__mechanism_dict[eps]
            except KeyError:
                pass
        if self.__eps == eps:
            return self.__mechanism
        self.__eps = eps
        self.__mechanism = random_response_mechanism(alphabet_size=self.STATE_COUNT, eps=eps)
        self.__mechanism_dict[eps] = self.__mechanism
        return self.__mechanism

    def get_name(self):
        return "Random Response"
