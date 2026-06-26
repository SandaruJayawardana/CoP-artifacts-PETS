
from mechanisms.CoP.convex_optimizer import *
from mechanisms.privacy_mechanism import *
from utils.util_functions import validate_alphabet, validate_distribution, validate_err_matrix

class Optimized_Randomized_Response(Privacy_Mechanism):
    
    def __init__(self, prior_dist, STATE_COUNT, INPUT_ALPHABET = [], normalized_objective_err_matrix = [], 
                    is_dist_preserve = True, ALPHA=0.0001, cache_on = False):
        
        self.STATE_COUNT = STATE_COUNT
        
        if len(INPUT_ALPHABET) != STATE_COUNT:
            self.INPUT_ALPHABET = str(range(STATE_COUNT))
        elif validate_alphabet(INPUT_ALPHABET):
            self.INPUT_ALPHABET = INPUT_ALPHABET
        
        if validate_distribution(prior_dist, self.STATE_COUNT):
            self.prior_dist = prior_dist

        if validate_err_matrix(normalized_objective_err_matrix, self.STATE_COUNT):
            self.normalized_objective_err_matrix = normalized_objective_err_matrix

        self.__mechanism = 0
        self.__eps = -1
        self.cache_on = cache_on
        self.__mechanism_dict = {}
        self.optimizer = Optimizer(self.prior_dist, self.normalized_objective_err_matrix, STATE_COUNT, is_dist_preserve, ALPHA)

    def get_mechanism(self, eps):
        if self.cache_on:
            try:
                return self.__mechanism_dict[eps]
            except KeyError:
                pass
        if self.__eps == eps:
            return self.__mechanism
            
        self.__eps = eps
        self.__mechanism = self.optimizer.get_mechanism(eps=eps)
        
        self.__mechanism_dict[eps] = self.__mechanism
        return self.__mechanism

    def get_name(self):
        return "Optimized Random Response"
