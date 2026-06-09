import pickle
from utils.mutual_information import pointwise_mutual_information
from utils.cmf_pmf_cal import get_cmf_pmf_dict_with_alphabet

FILE_LOCATION = '/home/sjay9734/CPM-publish/saved_metadata'

def get_pmi_dict(data, dataset, name):
    
    try:
        original_data = data 

        with open(f'{FILE_LOCATION}/{dataset}/{name}_pmi.pkl', 'rb') as file:
            PMI_dict, _ = pickle.load(file)

    except:
        original_data = data
        COLUMNS = original_data.columns

        PMF_dict, alphabet_dict = get_cmf_pmf_dict_with_alphabet(data=original_data, dataset=dataset, name=name, is_pmf=True)
        PMI_dict = pointwise_mutual_information(PMF_dict=PMF_dict, COLUMNS=COLUMNS, alphabet_dict=alphabet_dict)
        
        with open(f'{FILE_LOCATION}/{dataset}/{name}_pmi.pkl', 'wb') as file:
            pickle.dump((PMI_dict, COLUMNS), file)

    return PMI_dict
