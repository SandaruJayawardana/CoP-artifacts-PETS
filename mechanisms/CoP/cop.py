import numpy as np
import random, time


from mechanisms.privacy_mechanism import Privacy_Mechanism
from mechanisms.CoP.optimized_rr import Optimized_Randomized_Response
from utils.normalize_error_matrix import *

class CoP_Mechanism(Privacy_Mechanism):

    def __init__(self, ordered_attribute_list, alphabet_dict, THR = 0.6, THR_mi = 0, mi_dict= {}, pmi_dict = {}, CMF_dict = {}, err_type = "0_1", ALPHA=0.1, cache_on=True, numerical_list = {}, mixed_errors = False):


        self.ordered_attribute_list = ordered_attribute_list
        self.THR_mi= THR_mi
        self.alphabet_dict = alphabet_dict
        self.mechanism_list = []
        self.err_type = err_type
        self.THR = THR
        self.pmi_dict = pmi_dict
        self.CMF_dict = CMF_dict
        self.__eps = 0
        self.cache_on= cache_on
        self.mechanism_dict_individual = {}
        self.mechanism_dict = {}
        self.ALPHA = ALPHA
        self.mixed_errors = mixed_errors
        self.numerical_list = numerical_list

        if len(list(mi_dict.keys()))==0:
            self.mi_dict = {}
            for xk in ordered_attribute_list:
                for xl in ordered_attribute_list:
                    if xk == xl:
                        continue
                    key_ = xk + " " + xl
                    self.mi_dict[key_] = 1
        else:
            self.mi_dict = mi_dict

        self.get_mechanism()
    
    def traverse_all(self, attr, actual_dict, perturbed_dict, eps, perturbed_attribute_list, visited_set):
        perturbed_val = perturbed_dict[attr]
        
        for child_attr, max_prob, mechanism in self.mechanism_dict[attr][perturbed_val]:

            if child_attr in perturbed_attribute_list: #visited_set:
                continue
            
            perturbed_val = mechanism.gen_random_output(actual_value=actual_dict[child_attr], eps=eps)[0]

            perturbed_attribute_list.append(child_attr)

            perturbed_dict[child_attr] = perturbed_val

        return perturbed_dict, perturbed_attribute_list

    def create_optimal_mechanism_list(self):
        node_count = 0
        for index_xk, xk in enumerate(self.ordered_attribute_list):
            # init_key = xk + " " + self.ordered_attribute_list[(index_xk + 1) % attribute_count]
            self.mechanism_dict[xk] = {}
            xk_alphabets = self.alphabet_dict[xk]
            alpha = (1/len(xk_alphabets))*self.ALPHA
            # print("xk ", xk, xk_alphabets)
            for xk_alphabet_index, xk_alphabet in enumerate(xk_alphabets):
                self.mechanism_dict[xk][xk_alphabet] = []

                for index_z, z in enumerate(self.ordered_attribute_list):
                    if xk == z:
                        continue

                    key_ = xk + " " + z
                    CMF = self.CMF_dict[key_][xk_alphabet_index, :]
                    
                    # print("key_ ", key_, "CMF", CMF, state_count, self.CMF_dict[key_][xk_alphabet_index, :])
                    states = self.alphabet_dict[z]
                    state_count = len(states)
                    PMI = self.pmi_dict[key_][xk_alphabet_index, :]
                    max_cmf = max(PMI) # max_cmf = max(PMI)
                    if np.isnan(CMF[0]):
                        CMF = np.ones((state_count))/state_count

                    if max_cmf > self.THR and  self.mi_dict[key_] > self.THR_mi:
                        normalize_error_matrix = Normalize_error_matrix(attribute_list=[z], alphabet=states, priority_dict={}, alphabet_dict={z: states}, err_type=self.err_type)
                        # normalize_error_matrix = Normalize_error_matrix(alphabet=states, err_type=self.err_type)
                        err_matrix = normalize_error_matrix.normalized_error_matrix
                        mechanism = Optimized_Randomized_Response(prior_dist=CMF, STATE_COUNT = state_count, INPUT_ALPHABET = states, normalized_objective_err_matrix = err_matrix, 
                                is_dist_preserve=True, ALPHA=alpha, cache_on=True)
                        self.mechanism_dict[xk][xk_alphabet].append((z, max_cmf, mechanism))
                        node_count += 1
        # print("node_count ", node_count)
        
        for index_xk, xk in enumerate(self.ordered_attribute_list):
            states = self.alphabet_dict[xk]
            state_count = len(states)

            normalize_error_matrix = Normalize_error_matrix(attribute_list=[xk], alphabet=states, priority_dict={}, alphabet_dict={xk: states}, err_type=self.err_type)
            # normalize_error_matrix = Normalize_error_matrix(alphabet=states, err_type=self.err_type)
            err_matrix = normalize_error_matrix.normalized_error_matrix
            mechanism = Optimized_Randomized_Response(prior_dist = np.ones((state_count))/state_count, STATE_COUNT = state_count, INPUT_ALPHABET = states, normalized_objective_err_matrix = err_matrix, 
                                is_dist_preserve=True, ALPHA=self.ALPHA, cache_on=True)
            self.mechanism_dict_individual[xk] = mechanism
        
        for key_1 in list(self.mechanism_dict.keys()):
            for key_2 in list(self.mechanism_dict[key_1].keys()):
                # print("b ", self.mechanism_dict[key_1][key_2])
                self.mechanism_dict[key_1][key_2] = sorted(self.mechanism_dict[key_1][key_2], key=lambda x: x[1], reverse=True)
                # print(self.mechanism_dict[key_1][key_2])



    def get_mechanism(self):
        if self.mechanism_list == []:
            self.create_optimal_mechanism_list()
        return self.mechanism_list

    def get_eps(self):
        return self.__eps
    
    def gen_random_output(self, actual_value, eps):
        actual_dict = dict({})
        perturbed_dict = dict({})
        perturbed_attribute_list = []
        attribute_set = set(self.ordered_attribute_list)
        unvisited_set = attribute_set
        visited_set = set({})
        
        for index_i, i in enumerate(self.ordered_attribute_list):
            actual_dict[i] = str(actual_value[index_i])

        perturbed_attribute_list_index = 0

        while len(unvisited_set) > 0:
            np.random.seed(int((time.time()*1000000)%1000000))

            attr = random.choice(list(unvisited_set))
            mechanism = self.mechanism_dict_individual[attr]
            perturbed_val = str(mechanism.gen_random_output(actual_value=actual_dict[attr], eps=eps)[0])
            visited_set.add(attr)
            perturbed_attribute_list.append(attr)
            perturbed_dict[attr] = perturbed_val

            # if perturbed_val != actual_dict[attr]:
            #     continue

            while (perturbed_attribute_list_index < len(perturbed_attribute_list)):
                attr = perturbed_attribute_list[perturbed_attribute_list_index]
                perturbed_dict, perturbed_attribute_list = self.traverse_all(attr, actual_dict, perturbed_dict, eps, perturbed_attribute_list, visited_set)
                perturbed_attribute_list_index += 1
                visited_set.add(attr)

            
            unvisited_set = attribute_set - visited_set
        
        perturbed_list = []
        for i in self.ordered_attribute_list:
            perturbed_list.append(perturbed_dict[i])
        return perturbed_list
    
    def get_expected_utility_error(self, eps, input_probability = None):
        return np.sum(input_probability * np.sum(self.get_mechanism(eps) * self.normalized_objective_err_matrix, axis= 1))

    def get_name(self):
        return "CoP"
