import numpy as np
import pickle

def compute_H(G, G_prime, eps):
    Q = G/(G_prime + 1e-8)
    sorted_indices = np.argsort(Q)[::-1]
    A = 0
    B = 0
    for i in sorted_indices:
        if np.isnan(Q[i]) or Q[i] > (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1)):
            A += G[i]
            B += G_prime[i]
        else:
            break
    return (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1))

def Algo1_updated_privacy_leakage(eps, CMF_list):
    EPS = np.exp(eps)
    num_attr = len(CMF_list)
    num_rows, num_columns = np.shape(CMF_list[0])
    max_val = 0
    if num_rows == 1:
        return eps, 0
    
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
                
            if max_val < L:
                max_val = L
    leakage = np.log(max_val) + eps
    return leakage

def get_xk(key_):
    return key_.split(" ")[0]

def get_xl(key_):
    return key_.split(" ")[1]

def thoeretical_privacy_leakage(eps, CMF_dict, COLUMNS):

    cmf_keys = list(CMF_dict.keys())
    max_leakage = 0
    leakage_dict = {}
    for attr in COLUMNS:
        CMF_list = []
        for cmf_key in cmf_keys:
            if get_xk(cmf_key) == attr and get_xl(cmf_key) in COLUMNS:
                CMF_list.append(CMF_dict[cmf_key])
            
        leakage = Algo1_updated_privacy_leakage(CMF_list=CMF_list, eps=eps)
        leakage_dict[attr] = leakage
        if max_leakage < leakage:
            max_leakage = leakage

    return max_leakage, leakage_dict
