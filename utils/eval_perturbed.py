import pickle, math
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor
from utils.mutual_information import multivariate_mutual_information, exact_tot_cil
from utils.empirical_privacy import Empirical_Privacy
from utils.util_functions import _0_1_cal

from utils._freq_est import cpm_freq_est, krr_debias_freq_est
from utils._mean_est import cpm_mean_est, mean_from_krr

from utils.mutual_nsb import *

import numpy as np

def krr_debias_marginal(y_col, m, eps, clip=True):
    """
    y_col: (n,) int in [0..m-1]  (perturbed reports)
    returns p_hat: (m,) estimated true frequencies
    """
    n = len(y_col)
    q_hat = np.bincount(y_col, minlength=m) / n

    ee = np.exp(eps)
    p_hat = ((ee + m - 1) * q_hat - 1.0) / (ee - 1.0)  # unbiased

    if clip:
        p_hat = np.maximum(p_hat, 0.0)
        s = p_hat.sum()
        if s > 0: p_hat /= s
    return p_hat

def all_marginals_krr(Y, m_list, eps_list, clip=True):
    """
    Y: (n,d) perturbed ints
    m_list: list of m_k
    eps_list: list of eps_k (or scalar eps)
    """
    d = Y.shape[1]
    if np.isscalar(eps_list):
        eps_list = [eps_list] * d
    return [krr_debias_marginal(Y[:,k], m_list[k], eps_list[k], clip=clip) for k in range(d)]


def metric_evaluate(parameters):
    original_data, perturbed_data, COLUMNS, number_of_copies, index_x_k, X_k, is_info_leakage, is_privacy_leakage, is_exact_cil, is_nsb_MI, K = parameters
    privacy_leakage = Empirical_Privacy(original_data=original_data, perturb_data=perturbed_data)

    # print("x_k", X_k, "number_of_copies ", number_of_copies)
    # print("original_data.values[:,index_x_k]", original_data.values[:,index_x_k])
    original_data_array = np.tile(np.array(original_data.values[:,index_x_k], dtype=np.int64), number_of_copies)
    perturbed_data_values = np.array(perturbed_data.values, dtype=np.int64)
    tot_CIL = 0
    DPL = 0
    normalised_CIL = 0
    normalised_DPL = 0
    # _0_1_error = 0
    tot_privacy_leakage = 0
    tot_MI = 0
    exact_tot_CIL = 0
    normalised_tot_CIL = 0
    
    if is_nsb_MI:
        tot_CIL = max(0, mutual_nsb_cal(xn=original_data_array, Yn=perturbed_data_values, xn_index=index_x_k, k=K))

    elif is_info_leakage:
        # print("original_data_array ", original_data_array.dtype, original_data_array)
        # print("perturbed_data.values ",perturbed_data_values.dtype, perturbed_data_values)
        tot_CIL, DPL, normalised_tot_CIL, normalised_DPL, tot_MI = multivariate_mutual_information(xn=original_data_array, Yn=perturbed_data_values, xn_index=index_x_k, is_tot_DPL=True)
    
    # if is_utility:
    #     # alphabet_size = len(np.unique(original_data.values[:,index_x_k]))
    #     _0_1_error = _0_1_cal(actual=original_data_array, perturbed=perturbed_data_values[:,index_x_k])
    if is_exact_cil:
        exact_tot_CIL = exact_tot_cil(xn=original_data_array, Yn=perturbed_data_values, xn_index=index_x_k)

    if is_privacy_leakage:
        tot_privacy_leakage = privacy_leakage.additional_privacy_leakage(X_k=[X_k], Z=COLUMNS, p_value_cal=False, num_surrogates=20, samples = 1000000000)[0]

    print("C", exact_tot_CIL, tot_CIL, DPL, normalised_tot_CIL, normalised_DPL, tot_privacy_leakage, tot_MI, X_k)

    return {"tot_CIL": tot_CIL, "DPL": DPL, "normalised_tot_CIL": normalised_tot_CIL, "normalised_DPL": normalised_DPL, "tot_privacy_leakage": tot_privacy_leakage, "tot_MI": tot_MI, "X_k": X_k, "exact_tot_CIL": exact_tot_CIL}

class Evaluate_Perturbed_Data:

    def __init__(self, original_data, number_of_copies, EPS_ARRAY, COLUMNS, perturbed_data_list, save_location, NUM_PROCESS = 10, is_l2 = False, is_mse = False, is_exact_cil = False, is_freq_est = False, is_debiased = False, is_attribute_wise_utility = False, alphabet_dict = {}, numerical = [], eps_list = []):
        self.number_of_copies = number_of_copies
        self.EPS_ARRAY = EPS_ARRAY
        self.COLUMNS = COLUMNS
        self.save_location = save_location
        self.NUM_PROCESS = NUM_PROCESS
        self.NUM_ATTRIBUTES = len(COLUMNS)
        self.is_l2 = is_l2
        self.encoders = {}
        self.original_data_encoded = original_data.astype(str) # pd.DataFrame(original_data)
        self.perturbed_data_list_encoded = []
        self.is_attribute_wise_utility = is_attribute_wise_utility
        self.alphabet_range = []
        self.alphabet_min = []
        self.alphabet_max = []
        self.attribute_count = len(COLUMNS)
        self.numerical = numerical
        self.is_mse = is_mse
        self.is_exact_cil = is_exact_cil
        self.is_freq_est = is_freq_est
        self.eps_list = eps_list
        self.is_debiased = is_debiased
        self.m_list = []

        if  is_l2 or is_mse or is_freq_est:
            if len(alphabet_dict.keys()) == 0:
                raise ValueError(f"Empty alphabet_dict {alphabet_dict}")
            encoder_dict = {}
            self.mapping_dict_dict = {}
            for attr in COLUMNS:
                mapping_dict = {}

                alphabet = alphabet_dict[attr]
                self.m_list.append(alphabet)
                try:
                    vals = []
                    for a in alphabet:
                        mapping_dict[a] = float(a)
                        vals.append(float(a))
                except:
                    vals = []
                    for index_a, a in enumerate(alphabet):
                        mapping_dict[a] = index_a + 1
                        vals.append(index_a + 1)
                max_alphabet = max(vals)
                min_alphabet = min(vals)
                self.alphabet_range.append(abs(max_alphabet - min_alphabet))
                self.alphabet_min.append(min_alphabet)
                self.alphabet_max.append(max_alphabet)
                self.mapping_dict_dict[attr] = mapping_dict

                # print("mapping_dict ", mapping_dict)
                # print("alphabet ", alphabet)

                encoder_dict[attr] = mapping_dict
                # print("self.original_data_encoded before ", attr, self.original_data_encoded.head(4)[attr])
                self.original_data_encoded[attr] = self.original_data_encoded[attr].map(mapping_dict).astype(float)
                # print("original_data_before_[attr]", attr, np.unique(original_data[attr].values))
                # print("self.original_data_encoded", attr, np.unique(self.original_data_encoded[attr]))

            for i_eps, eps in enumerate(EPS_ARRAY):

                perturbed_data = pd.DataFrame(perturbed_data_list[i_eps][COLUMNS])
                # print("before i_eps ", i_eps, perturbed_data.head(3))
                for col in perturbed_data.columns:
                    perturbed_data[col] = perturbed_data[col].map(encoder_dict[col]).fillna(-1).astype(float)
                    # print("eps ", eps, "perturbed_data[col]", col, np.unique(perturbed_data[col].values))
                # print("i_eps ", i_eps, perturbed_data.head(3))
                self.perturbed_data_list_encoded.append(perturbed_data)
            self.alphabet_range = np.array(self.alphabet_range)
            self.alphabet_min = np.array(self.alphabet_min)
            self.alphabet_max = np.array(self.alphabet_max)
            # print("self.original_data_encoded", self.original_data_encoded.head(5))
        else:

            for attr in COLUMNS:
                le = LabelEncoder()
                self.original_data_encoded[attr] = le.fit_transform(original_data[attr])
                self.encoders[attr] = le
            # print("COLUMNS", COLUMNS)
            # print("original", self.original_data_encoded)
            for i_eps, eps in enumerate(EPS_ARRAY):

                perturbed_data = pd.DataFrame(perturbed_data_list[i_eps][COLUMNS])

                for col in perturbed_data.columns:
                    perturbed_data[col] = self.encoders[col].transform(perturbed_data[col])
                print("i_eps ", i_eps, perturbed_data.head())
                self.perturbed_data_list_encoded.append(perturbed_data)

    def evaluate_perturbed_data(self, is_info_leakage = True, is_privacy_leakage = True, is_utility = True, is_nsb_MI = False, K = 1):

        for i_eps, eps in enumerate(self.EPS_ARRAY):
            print(" eps ", eps)
            
            inner_tot_privacy_leakage = []
            inner_tot_CIL = []
            inner_tot_DPL = []
            inner_tot_normalised_CIL = []
            inner_tot_normalised_DPL = []
            inner_tot_MI = []
            inner_tot_exact_MI = []

            perturbed_data = self.perturbed_data_list_encoded[i_eps]

            last_index = 0
            while last_index < self.NUM_ATTRIBUTES:
                sub_attribute_list = [] 
                for i in range(self.NUM_PROCESS):
                    actual_index = last_index + i
                    if actual_index >= self.NUM_ATTRIBUTES:
                        break
                    sub_attribute_list.append(self.COLUMNS[actual_index])

                num_processes = min(self.NUM_PROCESS, len(sub_attribute_list))
                                    
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = [executor.submit(metric_evaluate, (self.original_data_encoded, perturbed_data, self.COLUMNS, self.number_of_copies, (j + last_index), sub_attribute_list[j], is_info_leakage, is_privacy_leakage, self.is_exact_cil, is_nsb_MI, K)) for j in range(num_processes)]
                    results_ = [future.result() for future in futures]

                for k in sub_attribute_list:
                    for r in results_:
                        if r["X_k"] == k:
                            inner_tot_privacy_leakage.append(r["tot_privacy_leakage"])
                            inner_tot_CIL.append(r["tot_CIL"])
                            inner_tot_exact_MI.append(r["exact_tot_CIL"])
                            inner_tot_DPL.append(r["DPL"])
                            inner_tot_normalised_CIL.append(r["normalised_tot_CIL"])
                            inner_tot_normalised_DPL.append(r["normalised_DPL"])
                            inner_tot_MI.append(r["tot_MI"])
                            break
                last_index += len(sub_attribute_list)
            
            # outter_tot_privacy_leakage.append(inner_tot_privacy_leakage)
            # outter_tot_CIL.append(inner_tot_CIL)
            # outter_0_1_loss.append(inner_0_1_loss)
            # outter_tot_DPL.append(inner_tot_DPL)

            inner_tot_privacy_leakage = np.array(inner_tot_privacy_leakage)
            inner_tot_CIL = np.array(inner_tot_CIL)
            inner_tot_DPL = np.array(inner_tot_DPL)
            inner_tot_normalised_CIL = np.array(inner_tot_normalised_CIL)
            inner_tot_normalised_DPL = np.array(inner_tot_normalised_DPL)
            inner_tot_MI = np.array(inner_tot_MI)
            inner_tot_exact_MI = np.array(inner_tot_exact_MI)

            if self.is_freq_est:

                X_array = self.original_data_encoded.values
                Y_array = perturbed_data.head(np.shape(X_array)[0]).values
                print("X_array ", X_array.shape, np.unique(X_array))
                print("Y_array ", np.unique(Y_array))

                X_array = np.asarray(X_array, dtype=int)
                Y_array = np.asarray(Y_array, dtype=int)

                for col_ in range(X_array.shape[1]):
                    le = LabelEncoder()
                    X_array[:, col_] = le.fit_transform(X_array[:,col_])
                    Y_array[:, col_] = le.transform(Y_array[:,col_])
                print("X_array ", X_array.shape, np.unique(X_array))
                print("Y_array ", np.unique(Y_array))
                if self.is_debiased:
                    freq_est = krr_debias_freq_est(Y=Y_array, X=X_array, m_list=self.m_list, eps_list=self.eps_list[i_eps])
                else:
                    freq_est = cpm_freq_est(Y=Y_array, X=X_array, m_list=self.m_list)

                if self.is_debiased:
                    mean_est = mean_from_krr(Y=Y_array, X=X_array, m_list=self.m_list, eps_list=self.eps_list[i_eps])
                else:
                    mean_est = cpm_mean_est(Y=Y_array, X=X_array, m_list=self.m_list)

                with open(f"{self.save_location}_freq_est_{eps}.pkl", 'wb') as file:
                    print("freq_est", freq_est, "mean_est", mean_est)
                    pickle.dump({"freq_est": freq_est, "mean_est": mean_est, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)

            if is_privacy_leakage:
                with open(f"{self.save_location}_privacy_leakage_evaluation_eps_{eps}.pkl", 'wb') as file:
                    pickle.dump({"privacy_leakage": inner_tot_privacy_leakage, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)

            if is_info_leakage:
                with open(f"{self.save_location}_info_leakage_evaluation_eps_{eps}.pkl", 'wb') as file:
                    pickle.dump({"total_CIL": inner_tot_CIL, "total_DPL": inner_tot_DPL, "inner_tot_normalised_DPL": inner_tot_normalised_DPL, 
                                 "inner_tot_normalised_CIL": inner_tot_normalised_CIL, "tot_MI": inner_tot_MI, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)
            if self.is_exact_cil:
                with open(f"{self.save_location}_total_exact_CIL_{eps}.pkl", 'wb') as file:
                    pickle.dump({"total_exact_CIL": inner_tot_exact_MI, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)

            if is_utility:
                # row_count = perturbed_data.shape[0]
                # utility_list = []
                # for row_num in range(row_count):
                #     utility_list.append(_0_1_cal(self.original_data_encoded.values[row_num,:], perturbed_data.values[row_num,:]))
                # utility_list = np.mean(np.array(utility_list))
                
                print("perturbed_data.values", np.shape(perturbed_data.values))
                original_data_array = np.tile(np.array(self.original_data_encoded.values, dtype=np.int64), (self.number_of_copies, 1))
                print("original_data_array", np.shape(original_data_array))
                
                if self.is_attribute_wise_utility:
                    errors = perturbed_data.values != original_data_array
                    utility = np.mean(errors, axis=0)
                elif self.is_l2:
                    # for attr in self.COLUMNS:
                    #     perturbed_data[attr] = perturbed_data[attr].map(self.mapping_dict_dict[attr]).astype(float)
                        
                    perturbed_data_values = (perturbed_data.values - self.alphabet_min)/self.alphabet_range
                    original_data_array = (original_data_array - self.alphabet_min)/self.alphabet_range
                    errors = np.linalg.norm(perturbed_data_values - original_data_array, axis=1)
                    utility = np.mean(errors/np.sqrt(self.attribute_count))
                    print("l2 utility ", utility, np.max(original_data_array), np.min(original_data_array), np.max(perturbed_data_values), np.min(perturbed_data_values))
                elif self.is_mse:

                    # for attr in self.COLUMNS:
                    #     perturbed_data[attr] = perturbed_data[attr].map(self.mapping_dict_dict[attr]).astype(float)
                    errors_cat = (1-self.numerical)*(perturbed_data.values != original_data_array)
                    perturbed_data_values = (perturbed_data.values - self.alphabet_min)/self.alphabet_range
                    original_data_array = (original_data_array - self.alphabet_min)/self.alphabet_range
                    errors_numerical = self.numerical*np.square(perturbed_data_values - original_data_array) + errors_cat
                    
                    # print("(perturbed_data.values, original_data_array)", perturbed_data_values, original_data_array)
                    # print("(perturbed_data.values != original_data_array)", (perturbed_data_values != original_data_array))
                    # print("(1-self.numerical)*(perturbed_data.values != original_data_array)", (1-self.numerical)*(perturbed_data_values != original_data_array))
                    
                    
                    utility = np.mean(np.mean(errors_numerical, axis=1))
                    # print("0-1 loss", np.mean(np.mean(errors_cat, axis=1)))
                    print("mse utility ", utility, np.max(original_data_array), np.min(original_data_array), np.max(perturbed_data_values), np.min(perturbed_data_values))
                else:
                    errors = perturbed_data.values != original_data_array
                    utility = np.mean(np.mean(errors, axis=1))
                    print("0-1 loss", utility)
                with open(f"{self.save_location}_utility_evaluation_eps_{eps}.pkl", 'wb') as file:
                    print("Saved ", f"{self.save_location}_utility_evaluation_eps_{eps}.pkl")
                    pickle.dump({"_0_1_loss": utility, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)
                

        # outter_tot_privacy_leakage = np.array(outter_tot_privacy_leakage)
        # outter_tot_CIL = np.array(outter_tot_CIL)
        # outter_0_1_loss = np.array(outter_0_1_loss)
        # outter_tot_DPL = np.array(outter_tot_DPL)

        # if is_privacy_leakage:
        #     with open(f"{self.save_location}_privacy_leakage_evaluation.pkl", 'wb') as file:
        #         pickle.dump({"privacy_leakage": outter_tot_privacy_leakage, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)

        # if is_info_leakage:
        #     with open(f"{self.save_location}_info_leakage_evaluation.pkl", 'wb') as file:
        #         pickle.dump({"total_CIL": outter_tot_CIL, "total_DPL": outter_tot_DPL, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)

        # if is_utility:
        #     with open(f"{self.save_location}_utility_evaluation.pkl", 'wb') as file:
        #         pickle.dump({"_0_1_loss": outter_0_1_loss, "COLUMNS": self.COLUMNS, "EPS_LIST": self.EPS_ARRAY}, file)
            