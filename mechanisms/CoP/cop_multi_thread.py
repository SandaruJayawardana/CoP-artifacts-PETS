import numpy as np
import random, time
from concurrent.futures import ProcessPoolExecutor

from mechanisms.privacy_mechanism import Privacy_Mechanism
from mechanisms.CPM.optimised_rr import Optimized_Randomized_Response
from utils.normalize_error_matrix import *

def gen_mechanisms_for_attribute(parameters):
    xk, ordered_attribute_list, alphabet_dict, CMF_dict, pmi_dict, THR, mechanism_gen_time, mi_dict, THR_mi = parameters

    mechanism_dict = {}
    xk_alphabets = alphabet_dict[xk]
            
    # print("xk ", xk, xk_alphabets)
    for xk_alphabet_index, xk_alphabet in enumerate(xk_alphabets):
        mechanism_dict[xk_alphabet] = []

        for index_z, z in enumerate(ordered_attribute_list):
            if xk == z:
                continue

            key_ = xk + " " + z
            CMF = CMF_dict[key_][xk_alphabet_index, :]
                    
            states = alphabet_dict[z]
            state_count = len(states)
            PMI = pmi_dict[key_][xk_alphabet_index, :]
            max_cmf = max(PMI) # max_cmf = max(PMI)
            if np.isnan(CMF[0]):
                CMF = np.ones((state_count))/state_count

            if max_cmf > THR and  mi_dict[key_] > THR_mi:
                normalize_error_matrix = Normalize_error_matrix(attribute_list=[z], alphabet=states, priority_dict={}, alphabet_dict={z: states}, err_type="0_1")
                err_matrix = normalize_error_matrix.normalized_error_matrix
                mechanism = Optimized_Randomized_Response(prior_dist=CMF, STATE_COUNT = state_count, INPUT_ALPHABET = states, normalized_objective_err_matrix = err_matrix, 
                                is_dist_preserve=True, ALPHA=0.001, cache_on=True)
                if mechanism_gen_time:
                    mechanism.gen_random_output(actual_value=states[0], eps=1)
                mechanism_dict[xk_alphabet].append((z, max_cmf, mechanism))
 
    states = alphabet_dict[xk]
    state_count = len(states)

    normalize_error_matrix = Normalize_error_matrix(attribute_list=[z], alphabet=states, priority_dict={}, alphabet_dict={z: states}, err_type="0_1")
    err_matrix = normalize_error_matrix.normalized_error_matrix
    mechanism = Optimized_Randomized_Response(prior_dist = np.ones((state_count))/state_count, STATE_COUNT = state_count, INPUT_ALPHABET = states, normalized_objective_err_matrix = err_matrix, 
                                is_dist_preserve=False, ALPHA=0.001, cache_on=True)
    mechanism_dict_individual = mechanism
    if mechanism_gen_time:
        mechanism.gen_random_output(actual_value=states[0], eps=1)
    return xk, mechanism_dict, mechanism_dict_individual

class CPM_Mechanism_Multithread(Privacy_Mechanism):

    def __init__(self, ordered_attribute_list, alphabet_dict, THR = 0.6, pmi_dict = {}, THR_mi = 0, mi_dict= {}, CMF_dict = {}, err_type = "0_1", ALPHA=0.01, cache_on=True, NUM_PROCESS = 10):

        self.ordered_attribute_list = ordered_attribute_list
        self.NUM_ATTRIBUTES = len(ordered_attribute_list)
        self.NUM_PROCESS = NUM_PROCESS
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
        self.THR_mi = THR_mi

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

            if child_attr in visited_set:
                continue
            
            perturbed_val = mechanism.gen_random_output(actual_value=actual_dict[child_attr], eps=eps)[0]

            perturbed_attribute_list.append(child_attr)

            perturbed_dict[child_attr] = perturbed_val

        return perturbed_dict, perturbed_attribute_list

    def create_optimal_mechanism_list(self, mechanism_gen_time = True):
        last_index = 0
        actual_index = 0
        while last_index < self.NUM_ATTRIBUTES:
                sub_attribute_list = [] 
                for i in range(self.NUM_PROCESS):
                    actual_index = last_index + i
                    if actual_index >= self.NUM_ATTRIBUTES:
                        break
                    sub_attribute_list.append(self.ordered_attribute_list[actual_index])

                num_processes = min(self.NUM_PROCESS, len(sub_attribute_list))
                                    
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    # xk, ordered_attribute_list, alphabet_dict, CMF_dict, pmi_dict, THR, mechanism_gen_time
                    futures = [executor.submit(gen_mechanisms_for_attribute, (sub_attribute_list[j], self.ordered_attribute_list, self.alphabet_dict, self.CMF_dict, self.pmi_dict, self.THR, mechanism_gen_time, self.mi_dict, self.THR_mi)) for j in range(num_processes)]
                    results_ = [future.result() for future in futures]

                for k in sub_attribute_list:
                    for r in results_:
                        if r[0] == k:
                            self.mechanism_dict[k] = r[1]
                            self.mechanism_dict_individual[k] = r[2]
                            break
                last_index += len(sub_attribute_list)
            
        for key_1 in list(self.mechanism_dict.keys()):
            for key_2 in list(self.mechanism_dict[key_1].keys()):
                # print("b ", self.mechanism_dict[key_1][key_2])
                self.mechanism_dict[key_1][key_2] = sorted(self.mechanism_dict[key_1][key_2], key=lambda x: x[1], reverse=True)
                # print(self.mechanism_dict[key_1][key_2])

        # for attr in self.ordered_attribute_list:
        #     print(attr, self.mechanism_dict[attr])
            

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
        return "CPM"
