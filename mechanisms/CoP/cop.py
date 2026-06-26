
"""
Implementation of Coordinated perturbation mechanism (CoP)
==========================================================

Paper - CoP: Coordinated Perturbation for Controlled Disclosure 
Under Local Differential Privacy (PETS 26)

This module implements the Coordinated Perturbation mechanism (CoP) for
locally differentially private perturbation of multidimensional user data.

The class `CoP_Mechanism` builds attribute-wise perturbation mechanisms using
prior dependency information, including mutual information (MI), pointwise
mutual information (PMI), and conditional probability matrices (CMFs). These
dependencies are used to coordinate the perturbation process across attributes
so that strongly dependent attributes can be perturbed in a dependency-aware
manner.

The class supports the following main steps:

1. Dependency-aware mechanism construction:
   For each attribute value, the class identifies dependent attributes using
   MI and PMI thresholds. For selected dependencies, it constructs atomic
   mechanisms (AMs).

2. Coordinated perturbation (Stochastic):
   Given a multidimensional input record, the class randomly selects unvisited
   attributes, perturbs them, and then traverses their dependent attributes to
   generate coordinated perturbed outputs.

## Main Methods

CoP_Mechanism(...)
Initializes the CoP mechanism with ordered attributes, alphabets,
dependency information, conditional probability matrices, and mechanism
design parameters.

create_optimal_mechanism_list()
Constructs dependency-aware optimized randomized response mechanisms and
individual attribute-wise mechanisms.

traverse_all(attr, actual_dict, perturbed_dict, eps,
perturbed_attribute_list, visited_set)
Perturbs dependent child attributes associated with the perturbed value of
the given attribute.

gen_random_output(actual_value, eps)
Generates a perturbed multidimensional output record under the given
privacy budget.

get_name()
Returns the mechanism name.

## Inputs

ordered_attribute_list : list
Ordered list of attribute names. The input and output records follow this
order.

alphabet_dict : dict
Dictionary mapping each attribute name to its possible values.

PMI_thr : float, optional
Threshold used to select strong pointwise dependencies.

I_thr : float, optional
Threshold used to select strong mutual-information-based dependencies.

mi_dict : dict, optional
Dictionary containing pairwise mutual information values.

pmi_dict : dict, optional
Dictionary containing PMI values for attribute-value dependencies.

CMF_dict : dict, optional
Dictionary containing conditional probability matrices for attribute pairs.

err_type : str, optional
Type of error metric used to construct the normalized utility/error matrix.

ALPHA : float, optional
Constraint parameter used in optimized randomized response mechanisms.

cache_on : bool, optional
Whether to cache optimized mechanisms.

numerical_list : dict, optional
Dictionary indicating numerical attributes, if any.

mixed_errors : bool, optional
Whether mixed error types are used.

## Outputs

gen_random_output(actual_value, eps) returns a list containing the perturbed
attribute values in the same order as `ordered_attribute_list`.

## Author

Sandaru Jayawardana

"""

import numpy as np
import random, time


from mechanisms.privacy_mechanism import Privacy_Mechanism
from mechanisms.CoP.optimized_rr import Optimized_Randomized_Response
from utils.normalize_error_matrix import *
from utils.per_attribute_privacy_budget_compute import get_per_attribute_privacy_budget

class CoP_Mechanism(Privacy_Mechanism):

    def __init__(self, ordered_attribute_list, alphabet_dict, PMI_thr = 0.1, I_thr = 0.05, mi_dict= {}, pmi_dict = {}, CMF_dict = {}, err_type = "0_1", ALPHA=0.1, cache_on=True, numerical_list = {}, mixed_errors = False):
        self.ordered_attribute_list = ordered_attribute_list
        self.I_thr= I_thr
        self.alphabet_dict = alphabet_dict
        self.mechanism_list = []
        self.err_type = err_type
        self.PMI_thr = PMI_thr
        self.pmi_dict = pmi_dict
        self.CMF_dict = CMF_dict
        self.eps_attribute = 0
        self.eps_dict = {}
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

            if child_attr in perturbed_attribute_list:
                continue
            
            perturbed_val = mechanism.gen_random_output(actual_value=actual_dict[child_attr], eps=eps)[0]

            perturbed_attribute_list.append(child_attr)

            perturbed_dict[child_attr] = perturbed_val

        return perturbed_dict, perturbed_attribute_list

    def create_optimal_mechanism_list(self):
        node_count = 0
        for xk in self.ordered_attribute_list:
            self.mechanism_dict[xk] = {}
            xk_alphabets = self.alphabet_dict[xk]
            alpha = (1/len(xk_alphabets))*self.ALPHA
            for xk_alphabet_index, xk_alphabet in enumerate(xk_alphabets):
                self.mechanism_dict[xk][xk_alphabet] = []

                for z in self.ordered_attribute_list:
                    if xk == z:
                        continue

                    key_ = xk + " " + z
                    CMF = self.CMF_dict[key_][xk_alphabet_index, :]
                    states = self.alphabet_dict[z]
                    state_count = len(states)
                    PMI = self.pmi_dict[key_][xk_alphabet_index, :]
                    max_cmf = max(PMI)
                    if np.isnan(CMF[0]):
                        CMF = np.ones((state_count))/state_count

                    if max_cmf > self.I_thr and  self.mi_dict[key_] > self.PMI_thr:
                        normalize_error_matrix = Normalize_error_matrix(attribute_list=[z], alphabet=states, priority_dict={}, alphabet_dict={z: states}, err_type=self.err_type)
                        err_matrix = normalize_error_matrix.normalized_error_matrix
                        mechanism = Optimized_Randomized_Response(prior_dist=CMF, STATE_COUNT = state_count, INPUT_ALPHABET = states, normalized_objective_err_matrix = err_matrix, 
                                is_dist_preserve=True, ALPHA=alpha, cache_on=True)
                        self.mechanism_dict[xk][xk_alphabet].append((z, max_cmf, mechanism))
                        node_count += 1
        
        for xk in self.ordered_attribute_list:
            states = self.alphabet_dict[xk]
            state_count = len(states)

            normalize_error_matrix = Normalize_error_matrix(attribute_list=[xk], alphabet=states, priority_dict={}, alphabet_dict={xk: states}, err_type=self.err_type)
            err_matrix = normalize_error_matrix.normalized_error_matrix
            mechanism = Optimized_Randomized_Response(prior_dist = np.ones((state_count))/state_count, STATE_COUNT = state_count, INPUT_ALPHABET = states, normalized_objective_err_matrix = err_matrix, 
                                is_dist_preserve=True, ALPHA=self.ALPHA, cache_on=True)
            self.mechanism_dict_individual[xk] = mechanism
        
        for key_1 in list(self.mechanism_dict.keys()):
            for key_2 in list(self.mechanism_dict[key_1].keys()):
                self.mechanism_dict[key_1][key_2] = sorted(self.mechanism_dict[key_1][key_2], key=lambda x: x[1], reverse=True)

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

        if eps in self.eps_dict.keys():
            self.eps_attribute = self.eps_dict[eps]
        else:
            self.eps_attribute = get_per_attribute_privacy_budget(eps=eps, CMF_dict=self.CMF_dict, COLUMNS=self.ordered_attribute_list)
            self.eps_dict[eps] = self.eps_attribute

        for index_i, i in enumerate(self.ordered_attribute_list):
            actual_dict[i] = str(actual_value[index_i])

        perturbed_attribute_list_index = 0

        while len(unvisited_set) > 0:
            np.random.seed(int((time.time()*1000000)%1000000))

            attr = random.choice(list(unvisited_set))
            mechanism = self.mechanism_dict_individual[attr]
            perturbed_val = str(mechanism.gen_random_output(actual_value=actual_dict[attr], eps=self.eps_attribute)[0])
            visited_set.add(attr)
            perturbed_attribute_list.append(attr)
            perturbed_dict[attr] = perturbed_val

            while (perturbed_attribute_list_index < len(perturbed_attribute_list)):
                attr = perturbed_attribute_list[perturbed_attribute_list_index]
                perturbed_dict, perturbed_attribute_list = self.traverse_all(attr, actual_dict, perturbed_dict, self.eps_attribute, perturbed_attribute_list, visited_set)
                perturbed_attribute_list_index += 1
                visited_set.add(attr)

            unvisited_set = attribute_set - visited_set
        
        perturbed_list = []
        for i in self.ordered_attribute_list:
            perturbed_list.append(perturbed_dict[i])
        return perturbed_list

    def get_name(self):
        return "CoP (Single thread)"
