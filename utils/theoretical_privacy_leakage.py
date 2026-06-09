import numpy as np
import time 
import pickle

def compute_H(G, G_prime, eps):
    Q = G/G_prime # np.sort(G/G_prime)[::-1]
    sorted_indices = np.argsort(Q)[::-1]
    A = 0
    B = 0
    # count = []
    for i in sorted_indices:
        if np.isnan(Q[i]) or Q[i] > (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1)):
            # count.append(i)
            A += G[i]
            B += G_prime[i]
        else:
            break
    # A = sum(np.sort(G[count])[::-1])
    # B = sum(np.sort(G_prime[count])[::-1])
    return (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1))

def Algo1_updated_privacy_leakage(eps, CMF_list):
    # print("Algo 1 optimal")
    EPS = np.exp(eps)
    num_attr = len(CMF_list)
    num_rows, num_columns = np.shape(CMF_list[0])
    print("num_attr num_rows", num_attr, num_rows)
    max_val = 0
    if num_rows == 1:
        return eps, 0
    # start_time = time.time()
    
    for i in range(num_rows):
        for j in range(num_rows):
            if i == j:
                continue
            L = 1
            for k in range(num_attr):
                
                G = CMF_list[k][i,:]
                G_prime = CMF_list[k][j,:]
                h = compute_H(G, G_prime, eps)
                h = min(h, EPS)
                h = max(1, h)

                L *= h
            # print("np.log(L)", np.log(L), "L", L, "np.exp(eps)", np.exp(eps))
            if max_val < L:
                max_val = L
    leakage = np.log(max_val) + eps
    return leakage #, end_time - start_time

def get_xk(key_):
    return key_.split(" ")[0]

def get_xl(key_):
    return key_.split(" ")[1]

def thoeretical_privacy_leakage(eps, CMF_dict, COLUMNS, location, EPS, recompute=False):

    try:
        assert not(recompute), "Recomputing leakage"
        
        with open(f"{location}_privacy_leakage.pkl", 'rb') as file:
            load_data = pickle.load(file)
            max_leakage = load_data["max_leakage"]
            leakage_dict = load_data["leakage_dict"]

    except:

        cmf_keys = list(CMF_dict.keys())
        max_leakage = 0
        leakage_dict = {}
        for attr in COLUMNS:
            CMF_list = []
            for cmf_key in cmf_keys:
                if get_xk(cmf_key) == attr and get_xl(cmf_key) in COLUMNS:
                    print("xk ", get_xk(cmf_key), " xl ", get_xl(cmf_key), np.shape(CMF_dict[cmf_key]))
                    CMF_list.append(CMF_dict[cmf_key])
            
            leakage = Algo1_updated_privacy_leakage(CMF_list=CMF_list, eps=eps)
            
            leakage_dict[attr] = leakage

            if max_leakage < leakage:
                max_leakage = leakage

        with open(f"{location}_privacy_leakage.pkl", 'wb') as file:
            pickle.dump({"max_leakage": max_leakage, "leakage_dict": leakage_dict}, file)
            
    return max_leakage, leakage_dict
