import numpy as np
import pandas as pd
import pickle

class Evaluate_Perturbed_Data:

    def __init__(self, original_data, EPS_ARRAY, COLUMNS, perturbed_data_list, save_location, alphabet_dict = {}, numerical = [], eps_list = []):
        
        self.EPS_ARRAY = EPS_ARRAY
        self.save_location = save_location
        self.original_data_encoded = original_data.astype(str)
        self.perturbed_data_list_encoded = []
        self.alphabet_range = []
        self.alphabet_min = []
        self.alphabet_max = []
        self.numerical = numerical
        self.eps_list = eps_list
        self.m_list = []

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
            encoder_dict[attr] = mapping_dict
            self.original_data_encoded[attr] = self.original_data_encoded[attr].map(mapping_dict).astype(float)

        for i_eps, eps in enumerate(EPS_ARRAY):
            perturbed_data = pd.DataFrame(perturbed_data_list[i_eps][COLUMNS])
            for col in perturbed_data.columns:
                perturbed_data[col] = perturbed_data[col].map(encoder_dict[col]).fillna(-1).astype(float)
            self.perturbed_data_list_encoded.append(perturbed_data)
        self.alphabet_range = np.array(self.alphabet_range)
        self.alphabet_min = np.array(self.alphabet_min)
        self.alphabet_max = np.array(self.alphabet_max)


    def evaluate_perturbed_data(self):

        for i_eps, eps in enumerate(self.EPS_ARRAY):

            perturbed_data = self.perturbed_data_list_encoded[i_eps]

            original_data_array = np.array(self.original_data_encoded.values, dtype=np.int64)

            errors_cat = (1-self.numerical)*(perturbed_data.values != original_data_array)
            perturbed_data_values = (perturbed_data.values - self.alphabet_min)/self.alphabet_range
            original_data_array = (original_data_array - self.alphabet_min)/self.alphabet_range
            errors_numerical = self.numerical*np.square(perturbed_data_values - original_data_array) + errors_cat
            utility = np.mean(np.mean(errors_numerical, axis=1))

            with open(f"{self.save_location}_utility_evaluation_eps_{eps}.pkl", 'wb') as file:
                print("Saved ", f"{self.save_location}_utility_evaluation_eps_{eps}.pkl")
                pickle.dump({"_0_1_loss": utility, "EPS_LIST": self.EPS_ARRAY}, file)
                