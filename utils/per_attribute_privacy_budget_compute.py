from utils.theoretical_privacy_leakage import thoeretical_privacy_leakage

def get_per_attribute_privacy_budget(eps, CMF_dict, COLUMNS, e = 0.1):
    
    eps_hat = eps/len(COLUMNS)
    leakage, _ = thoeretical_privacy_leakage(eps=eps_hat, CMF_dict=CMF_dict, COLUMNS=COLUMNS)
    while leakage < eps:
        eps_hat += e
        leakage, _ = thoeretical_privacy_leakage(eps=eps_hat, CMF_dict=CMF_dict, COLUMNS=COLUMNS)
    
    return eps_hat
    