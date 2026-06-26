import numpy as np

def pmi_empirical(X_k_alphabet, X_l_alphabet, pmf):
    probability_dist = np.zeros((len(X_k_alphabet), len(X_l_alphabet)))

    for index_xk, _ in enumerate(X_k_alphabet):
        for index_xl, _ in enumerate(X_l_alphabet):
            probability_dist[index_xk][index_xl] += pmf[index_xk][index_xl]
            
    probability_dist /= np.sum(probability_dist)
    pmi_xk_xl = np.transpose(np.transpose(probability_dist/np.sum(probability_dist, axis=0))/np.sum(probability_dist, axis=1))
    
    return np.log2(pmi_xk_xl)

def pointwise_mutual_information(PMF_dict, COLUMNS, alphabet_dict):
    original_pmi_dict = {}
    for i in COLUMNS[:]: 
        X_k_alphabet = alphabet_dict[i]
        
        for j in COLUMNS[:]:
            if i == j:
                continue

            X_l_alphabet = alphabet_dict[j]

            key_ = str(i) + ' ' + str(j)
            pmf = PMF_dict[key_]
            original_pmi_dict[key_] = pmi_empirical(X_k_alphabet=X_k_alphabet, X_l_alphabet=X_l_alphabet, pmf=pmf)

    return original_pmi_dict

"""function infocontent(p)
Computes the Shannon information content for an outcome x of a random variable
X with probability p.

Inputs:
- p - probability to compute the Shannon info content for

Outputs:
- result - Shannon info content of the probability p

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""

def infocontent(p):

    return -np.log2(p)

"""function entropy(p)
Computes the Shannon entropy for a probability distribution p.

Inputs:
- p - (array which much sum to 1) - a probability distribution to compute the Shannon info content for

Outputs:
- result - Shannon entropy of the probability distribution p
 
Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def entropy(p):  
    # First make sure the array is now a numpy array
    if type(p) != np.array:
        p = np.array(p)

    # Should we check any potential error conditions on the input?
    if (abs(np.sum(p) - 1) > 0.00001):
        raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(p))
    
    # We need to take the expectation value over the Shannon info content at
    # p(x) for each outcome x:
    weightedShannonInfos = p*(infocontent(p))
    # nansum ignores the nans from calling infocontent(0), but we still get the warning if an entry in p is zero
    return np.nansum(weightedShannonInfos)

#################################
# End of module 1 functions
#################################

""" function entropyempirical(xn)
Computes the Shannon entropy over all outcomes x of a random variable
X from samples x_n.

Inputs:
- xn - samples of outcomes x.
   xn is a column vector, e.g. xn = [0;0;1;0;1;0;1;1;1;0] for a binary variable.

Outputs:
- result - Shannon entropy over all outcomes
- symbols - list of unique samples
- probabilities - probabilities for each sample

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def entropyempirical(xn):

    # First, error checking, and converting argument into standard form:    
    if type(xn) == list:
        xn = np.array(xn)
    if xn.ndim == 1:
        xn = np.reshape(xn,(len(xn), 1)) #reshaping our 1-dim vector to numpy format of a column vector
    [xnSamples,xnDimensions] = xn.shape
    
    # We need to work out the alphabet here.
    # The following returns a vector of the alphabet:    
    symbols = np.unique(xn, axis=0)
    
    # It would be faster to call:
    [symbols, counts] = np.unique(xn, axis=0, return_counts=True)
    
    # but we could count the samples manually below for instructive purposes:
    # Next we need to count the number of occurances of each symbol in 
    # the alphabet:
    # counts = []
    # for symbol in symbols:
    #    count = 0
    #    for row in xn:
    #        if (row==symbol).all():
    #            count += 1
    #    counts.append(count)
    # counts = np.array(counts);
    
    # Now normalise the counts into probabilities:
    probabilities = counts / xnSamples
    
    # Once we have the probabilities we can simply call our existing function:
    result = entropy(probabilities)
    
    return result, symbols, probabilities
    
""" function jointentropy(p)
Computes the joint Shannon entropy over all outcome vectors x of a vector
random variable X with probability matrix p(x) for each candidate outcome
vector x.

Inputs:
- p - probability distribution function over all outcome vectors x.
   p is a matrix over all combinations of the sub-variables of x,
where p(1,3) gives the probability of the first symbol of sub-variable
x1 co-occuring with the third symbol of sub-variable x2.
   E.g. p = [0.2, 0.3; 0.1, 0.4]. The sum over p must be 1.

Outputs:
- result - joint Shannon entropy of the probability distribution p

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def jointentropy(p):
    
    # Should we check any potential error conditions on the input?

    # We need to take the expectation value over the Shannon info content at
    #  p(x) for each outcome x in the joint PDF:
    # Hint: will your code for entropy(p) work, or can you alter it slightly
    #  to make it work?
    
    joint_entropy = entropy(p)
    
    return joint_entropy

""" function jointentropyempirical(xn, yn)
Computes the Shannon entropy over all outcome vectors x of a vector random
variable X from sample vectors x_n. User can call with two such arguments 
if they don't wish to join them outside of the call.

Inputs:
- xn - matrix of samples of outcomes x. May be a 1D vector of samples
    (in which case yn is also supplied), or
    a 2D matrix, where each row is a vector sample for a multivariate X
    (in which case yn is not supplied).
- yn - as per xn, except that yn is not required to be supplied (in which
 case the entropy is only calculated over the multivariate xn variable).

Outputs:
- result - joint Shannon entropy over all samples
- symbols - list of unique joint vector samples
- probabilities - probabilities for each joint symbol

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def jointentropyempirical(xn, yn=[]):
    
    # First, error checking, and converting argument into standard form:    
    xn = np.array(xn)
    # Convert to column vectors if not already:
    if xn.ndim == 1:
        xn = np.reshape(xn,(len(xn),1))
    yn = np.array(yn)
    if (yn.size > 0):
        # Convert to column vectors if not already:
        if yn.ndim == 1:
            yn = np.reshape(yn,(len(yn),1))
        [rx,cx] = xn.shape
        [ry,cy] = yn.shape
        # Check that their number of rows are the same:
        assert(rx == ry)
        # Now joint them up so we only need work with xn
        xn = np.concatenate((xn,yn), axis=1)
        
    # TRICK: Next combine the row vectors in each sample into a single 
    #  symbol (being the index from the symbols array,
    # so that we can simply compute entropy on that combined symbol
    [symbols, symbolIndexForEachSample] = np.unique(xn, axis=0, return_inverse=True)

    # And compute the entropy using our existing function:
    [result, symbols_of_indices, probabilities] = entropyempirical(symbolIndexForEachSample);

    # The order of symbols is the same as their order for the probabilities

    return result, symbols, probabilities

"""function conditionalentropy(p)

Computes the conditional Shannon entropy over all outcomes x of a random
variable X, given outcomes y of a random variable Y.
Probability matrix p(x,y) is given for each candidate outcome
(x,y).

Inputs:
- p - 2D probability distribution function over all outcomes (x,y).
   p is a matrix over all combinations of x and y,
where p(1,3) gives the probability of the first symbol of variable
x co-occuring with the third symbol of variable y.
   E.g. p = [0.2, 0.3; 0.1, 0.4]. The sum over p must be 1.

Outputs:
- result - conditional Shannon entropy of X given Y

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def conditionalentropy(p):
    
    # First make sure the array is now a numpy array
    if type(p) != np.array:
        p = np.array(p)

    # Should we check any potential error conditions on the input?
    # a. Should we check p is a matrix, not a vector?
    # Actually we won't since a vector would be valid if one variable only ever took one value.
    # b. Check that the probabilities normalise to 1:
    if (abs(np.sum(p) - 1) > 0.00001):
        raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(p))

    # We need to compute H(X,Y) - H(X):
    # 1. joint entropy: Can we re-use existing code?
    H_XY = jointentropy(p);
    # 2. marginal entropy of Y: Can we re-use existing code?
    #  But how to get p_y???
    p_y = p.sum(axis=0); # Since y changes along the columns, summing over the x's (dimension 0 argument in the sum) will just return p(y)
    H_Y = entropy(p_y);
    
    result = H_XY - H_Y;
    return result

"""function conditionalentropyempirical(xn, yn)
Computes the conditional Shannon entropy over all samples xn of a random
variable X, given samples yn of a random variable Y.

Inputs:
- xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate X.
- yn - matrix of samples of outcomes x. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate Y.
    Must have the same number of rows as X.

Outputs:
- result - conditional Shannon entropy of X given Y

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def conditionalentropyempirical(xn, yn):
    
    # First, error checking, and converting argument into standard form:    
    xn = np.array(xn)
    # Convert to column vectors if not already:
    if xn.ndim == 1:
        xn = np.reshape(xn,(len(xn),1))
    yn = np.array(yn)
    if yn.ndim == 1:
        yn = np.reshape(yn,(len(yn),1))
    [rx,cx] = xn.shape
    [ry,cy] = yn.shape

    # Should we check any potential error conditions on the input?
    # Check that their number of rows are the same:
    assert(rx == ry)
    
    # We need to compute H(X,Y) - H(X):
    # 1. joint entropy: Can we re-use existing code?
    (H_XY, xySymbols, xyProbs) = jointentropyempirical(xn, yn);
    # 2. marginal entropy of Y: Can we re-use existing code?
    (H_Y, ySymbols, yProbs) = entropyempirical(yn);
    
    result = H_XY - H_Y;
    return result

#################################
# End of module 2 functions
#################################

"""function mutualinformation(p)
Computes the mutual information over all outcomes x of a random
variable X with outcomes y of a random variable Y.
Probability matrix p(x,y) is given for each candidate outcome
(x,y).

Inputs:
- p - 2D probability distribution function over all outcomes (x,y).
   p is a matrix over all combinations of x and y,
where p(1,3) gives the probability of the first symbol of variable
x co-occuring with the third symbol of variable y.
   E.g. p = [0.2, 0.3; 0.1, 0.4]. The sum over p must be 1.

Outputs:
- result - mutual information of X with Y

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def mutualinformation(p):
    
    # First make sure the array is now a numpy array
    if type(p) != np.array:
        p = np.array(p)

    # Should we check any potential error conditions on the input?
    # a. Should we check p is a matrix, not a vector?
    # Actually we won't since a vector would be valid if one variable only ever took one value.
    # b. Check that the probabilities normalise to 1:
    if (abs(np.sum(p) - 1) > 0.00001):
        raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(p))

    # We need to compute H(X) + H(Y) - H(X,Y):
    # 1. joint entropy:
    H_XY = jointentropy(p)

    # 2. marginal entropy of X:
    # But how to get p_x???
    p_x = p.sum(axis=1); # Since x changes along the rows, summing over the y's (dimension 1 argument in the sum) will just return p(x)
    H_X = entropy(p_x);

    # 2. marginal entropy of Y:
    # But how to get p_y???
    p_y = p.sum(axis=0); # Since y changes along the columns, summing over the x's (dimension 0 argument in the sum) will just return p(y)
    H_Y = entropy(p_y);

    result = H_X + H_Y - H_XY
    
    return result

"""function mutualinformationempirical(xn,yn)
Computes the mutual information over all samples xn of a random
variable X with samples yn of a random variable Y.

Inputs:
- xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate X.
- yn - matrix of samples of outcomes x. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate Y.
    Must have the same number of rows as X.

Outputs:
- result - mutual information of X with Y
- xySymbols - list of unique joint vector samples
- xyProbs - probabilities for each joint symbol
- xSymbols - list of unique x samples
- xProbs - probabilities for each x symbol
- ySymbols - list of unique y samples
- yProbs - probabilities for y symbol

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def mutualinformationempirical(xn,yn):
    
    # First, error checking, and converting argument into standard form:    
    xn = np.array(xn)
    # Convert to column vectors if not already:
    if xn.ndim == 1:
        xn = np.reshape(xn,(len(xn),1))
    yn = np.array(yn)
    if yn.ndim == 1:
        yn = np.reshape(yn,(len(yn),1))
    [rx,cx] = xn.shape
    [ry,cy] = yn.shape

    # Should we check any potential error conditions on the input?
    # Check that their number of rows are the same:
    assert(rx == ry)

    # We need to compute H(X) + H(Y) - H(X,Y):
    # 1. joint entropy:
    (H_XY, xySymbols, xyProbs) = jointentropyempirical(xn, yn); # How to compute this empirically ...?
    # 2. marginal entropy of Y: (call 'joint' in case yn is multivariate)
    (H_Y, ySymbols, yProbs) = jointentropyempirical(yn)
    # 3. marginal entropy of X: (call 'joint' in case yn is multivariate)
    (H_X, xSymbols, xProbs) = jointentropyempirical(xn);
    
    result = H_X + H_Y - H_XY;
    # print("result ", result)
    return result, xySymbols, xyProbs, xSymbols, xProbs, ySymbols, yProbs

#################################
# End of module 3 functions
#################################

"""function conditionalmutualinformation(p)
Computes the mutual information over all outcomes x of a random
variable X with outcomes y of a random variable Y, conditioning on 
outcomes z of a random variable Z.
Probability matrix p(x,y,z) is given for each candidate outcome
(x,y,z).

Inputs:
- p - 3D probability distribution function over all outcomes (x,y,z).
   p is a matrix over all combinations of x and y and z,
where p(0,2,1) gives the probability of the first symbol of variable
x co-occuring with the third symbol of variable y and the second
symbol of z.
   The sum over p must be 1.
   E.g.:
     p[0,:,:] = [[0.114286, 0.171429], [0.057143, 0.228571]]
     p[1,:,:] = [[0.171429, 0.114286], [0.028571, 0.114286]]

Outputs:
- result - conditional mutual information of X with Y given Z
 
Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def conditionalmutualinformation(p):
    
    # First make sure the array is now a numpy array
    if type(p) != np.array:
        p = np.array(p)

    # Should we check any potential error conditions on the input?
    # a. Check that we have 3 dimensions. We allowed one dimension to be null
    #  for MI and conditional entropy, but for CMI we can't tell which is missing
    if (p.ndim != 3):
        raise Exception("Probability distribution must have 3 dimensions for CMI")
    # b. Check that the probabilities normalise to 1:
    if (abs(np.sum(p) - 1) > 0.00001):
        raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(p))

    # We need to compute H(X|Z) + H(Y|Z) - H(X,Y|Z).
    # But our conditional entropy calculator won't do H(X,Y|Z) since it doesn't accept a joint probability for X,Y.
    # So, easier to rewrite as:
    #  H(X,Z) - H(Z) + H(Y,Z) - H(Z) - H(X,Y,Z) + H(Z)
    #  = H(X,Z) - H(Z) + H(Y,Z) - H(X,Y,Z)
    
    # 1. joint entropy:
    H_XYZ = jointentropy(p)
    
    # 2. entropy of X,Z:
    #  But how to get p_xz???
    # Sum p over the y's (2nd dimension argument in the sum) will just return p(x,z) terms.
    p_xz = p.sum(axis=1)
    H_XZ = jointentropy(p_xz)

    # 3. entropy of Y,Z:
    #  But how to get p_yz???
    # Sum p over the x's (1st dimension argument in the sum) will just return p(y,z) terms.
    p_yz = p.sum(axis=0)
    H_YZ = jointentropy(p_yz)

    # 4. marginal entropy of Z:
    #  But how to get p_z???
    # Sum p_xz over the x's (1st dimension argument in the sum) will just return p(z) terms.
    p_z = p_xz.sum(axis=0)
    H_Z = jointentropy(p_z)
    
    result = H_XZ - H_Z + H_YZ - H_XYZ
    return result

"""function conditionalmutualinformationempirical(xn,yn,zn)
Computes the mutual information over all samples xn of a random
variable X with samples yn of a random variable Y, conditioning on 
samples zn of a random variable Z.

Inputs:
- xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate X.
- yn - matrix of samples of outcomes y. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate Y.
    Must have the same number of rows as X.
- zn - matrix of samples of outcomes z. May be a 1D vector of samples, or
    a 2D matrix, where each row is a vector sample for a multivariate Z
    which will be conditioned on.
    Must have the same number of rows as X.

Outputs:
- result - conditional mutual information of X with Y, given Z

Copyright (C) 2020-, Julio Correa, Joseph T. Lizier
Distributed under GNU General Public License v3
"""
def conditionalmutualinformationempirical(xn, yn, zn):
    
    # First, error checking, and converting argument into standard form:    
    xn = np.array(xn)
    # Convert to column vectors if not already:
    if xn.ndim == 1:
        xn = np.reshape(xn,(len(xn),1))
    yn = np.array(yn)
    if yn.ndim == 1:
        yn = np.reshape(yn,(len(yn),1))
    zn = np.array(zn)
    if zn.ndim == 1:
        zn = np.reshape(zn,(len(zn),1))
    [rx,cx] = xn.shape
    [ry,cy] = yn.shape
    [rz,cz] = zn.shape

    # Should we check any potential error conditions on the input?
    # Check that their number of rows are the same:
    assert(rx == ry)
    assert(rx == rz)

    # We need to compute H(X|Z) + H(Y|Z) - H(X,Y|Z):
    # 1. conditional joint entropy:
    H_XY_given_Z = conditionalentropyempirical(np.append(xn, yn, axis=1),zn); # How to compute this empirically ...?
    # 2. conditional entropy of Y:
    H_Y_given_Z = conditionalentropyempirical(yn,zn) # How to compute this empirically ...?
    # 3. conditional entropy of X:
    H_X_given_Z = conditionalentropyempirical(xn,zn) # How to compute this empirically ...?
    
    # Alternatively, note that we could compute I(X;Y,Z) - I(X;Z)
    
    result = H_X_given_Z + H_Y_given_Z - H_XY_given_Z;
    return result

# def get_combined_dataset(Yn, selected_indexes):
#     index_count = len(selected_indexes)

#     if index_count == 1:
#         return Yn[:,selected_indexes[0]]
    
#     new_data = []
#     data_count = len(Yn)
    
#     for i in range(data_count):
#         new_ = ""
#         for j in selected_indexes: #range(index_count):
#             new_ += str(Yn[i][j]) + " "
#         new_data.append(new_)
    
#     return np.array(new_data)

new_data_list_total = []
new_data_list_conditional = []

def get_combined_dataset_total(Yn):
    data_count = len(Yn)
    
    for i in range(data_count):
        new_data_list_total[i] += str(Yn[i]) + " "
    return np.array(new_data_list_total)

def get_combined_dataset_coditional(Yn):
    data_count = len(Yn)
    
    for i in range(data_count):
        new_data_list_conditional[i] += str(Yn[i]) + " "
    return np.array(new_data_list_conditional)

def multivariate_mutual_information(xn, Yn, xn_index = 0, is_tot_DPL = False):

    attribute_count = np.shape(Yn)[1]
    tot_CIL = 0 
    tot_MI = 0
    tot_normalised_CIL = 0
    # print("mi xn ", xn)
    for i in range(attribute_count):
        if i == xn_index:
            continue

        zn = Yn[:,xn_index]
        CIL = conditionalmutualinformationempirical(xn=xn, yn=Yn[:,i], zn=zn)
        MI = mutualinformationempirical(xn, Yn[:,i])[0]
        tot_MI += MI

        if CIL < 0.001:
            continue

        # entropy_1 = conditionalentropyempirical(xn, zn)
        # entropy_2 = conditionalentropyempirical(Yn[:,i], zn)
        # tot_normalised_CIL += CIL/(min(entropy_1, entropy_2))
        tot_CIL += CIL

    if is_tot_DPL:
        DPL = mutualinformationempirical(xn, Yn[:,xn_index])[0]
        entropy_xn = entropyempirical(xn=xn)[0]
        entropy_yn = entropyempirical(xn=Yn[:,xn_index])[0]
        normalised_DPL = DPL/(min(entropy_xn, entropy_yn))
        tot_normalised_CIL = min(CIL/entropy_xn, 1)
        return tot_CIL, DPL, tot_normalised_CIL, normalised_DPL, tot_MI
    return tot_CIL

def exact_tot_cil(xn, Yn, xn_index = 0):

    attribute_count = np.shape(Yn)[1]
    tot_CIL = 0 
    CIL_list = []
    index_list = [xn_index]

    for i in range(attribute_count):
        if i == xn_index:
            continue

        zn = Yn[:,index_list]
        CIL = conditionalmutualinformationempirical(xn=xn, yn=Yn[:,i], zn=zn)
        CIL_list.append(CIL)
        if CIL < 0.001:
            continue

        tot_CIL += CIL
        index_list.append(i)
    print(CIL_list)
    return tot_CIL


# def multivariate_mutual_information_original(xn, Yn, xn_index = 0, is_tot_DPL = False):
#     global new_data_list
#     new_data_list = []
#     for i in range(len(Yn)):
#         new_data_list.append("")

#     pair_wise_mi = {}
#     sorted_pair_wise_mi_list = []

#     # tot_DPL = 0

#     for index_i in range(np.shape(Yn)[1]):

#         i = Yn[:,index_i]
#         current_value = mutualinformationempirical(xn, i)[0]
#         pair_wise_mi[index_i] = current_value
#         # if index_i == xn_index:
#         #     tot_DPL = current_value
#         # print(mutualinformationempirical(xn, i)[3], mutualinformationempirical(xn, i)[5])

#         if len(sorted_pair_wise_mi_list) == 0:
#             sorted_pair_wise_mi_list = [index_i]
#         else:
#             is_added = False
#             for index_j, j in enumerate(sorted_pair_wise_mi_list):
#                 if pair_wise_mi[j] < current_value:
#                     sorted_pair_wise_mi_list = sorted_pair_wise_mi_list[:index_j] + [index_i] + sorted_pair_wise_mi_list[index_j:]
#                     is_added = True
#                     break
#             if not(is_added):
#                 sorted_pair_wise_mi_list.append(index_i)
    
#     S = []
#     for i in sorted_pair_wise_mi_list:
#         S.append(pair_wise_mi[i])
#     # print("sorted_pair_wise_mi_list ", sorted_pair_wise_mi_list, "xn_index", xn_index, " S ", S)
    
#     total_mutual_info = 0 # pair_wise_mi[sorted_pair_wise_mi_list[0]]

#     selected_indexes = []
#     conditional_mi_list = []

#     sum_normalised_entropy = 0

#     for index_i, i in enumerate(sorted_pair_wise_mi_list[1:]):

#         selected_indexes.append(sorted_pair_wise_mi_list[index_i])
#         zn_new_variable = Yn[:,xn_index] # get_combined_dataset(Yn=Yn, selected_indexes=selected_indexes) # Yn[:,selected_indexes[0]] # get_combined_dataset(Yn=Yn, selected_indexes=selected_indexes)
#         conditional_mi = conditionalmutualinformationempirical(xn=xn, yn=Yn[:,i], zn=zn_new_variable)
#         conditional_mi_list.append((sorted_pair_wise_mi_list[index_i], i, conditional_mi))
#         if conditional_mi < 0.001:
#             # total_mutual_info += conditional_mi
#             # break
#             continue
#         entropy_1 = conditionalentropy(np.concat((xn, zn_new_variable), axis=1))[0]
#         entropy_2 = conditionalentropy(np.concat((Yn[:,i], zn_new_variable), axis=1))[0]
#         sum_normalised_entropy += conditional_mi/(min(entropy_1, entropy_2))
#         total_mutual_info += conditional_mi
#     # print("conditional_mi_list ", conditional_mi_list)
#     # print("total_mutual_info ", total_mutual_info)


#     if is_tot_DPL:
#         entropy_xn = entropyempirical(xn=xn)
#         entropy_yn = entropyempirical(xn=Yn[:,xn_index])
#         normalised_DPL = pair_wise_mi[xn_index]/(min(entropy_xn, entropy_yn))
#         return total_mutual_info, pair_wise_mi[xn_index], sum_normalised_entropy/(len(sorted_pair_wise_mi_list)-1), normalised_DPL
#     return total_mutual_info
#     # return tot_DPL

def multivariate_mutual_information_logitudinal(xn, Yn_all):

    global new_data_list_total, new_data_list_conditional
    new_data_list_total = []
    for i in range(len(Yn_all[0])):
        new_data_list_total.append("")

    new_data_list_conditional = []
    for i in range(len(Yn_all[0])):
        new_data_list_conditional.append("")

    repetitions = np.shape(Yn_all)[0]
    selected_indexes_conditional = []
    selected_indexes = []

    total_mi = []
    conditional_mi = []
    individual_mi = []

    # print("Yn ", Yn_all)

    for i in range(repetitions):
        Yn = Yn_all[i]
        selected_indexes.append(i)
        yn_new_variable = get_combined_dataset_total(Yn=Yn) 
        # print("xn ", xn)
        # print("yn ", yn_new_variable)
        total_mi.append(mutualinformationempirical(xn=xn, yn=yn_new_variable)[0])
        individual_mi.append(mutualinformationempirical(xn=xn, yn=Yn)[0])

        if i != 0:
            selected_indexes_conditional.append(i-1)
            zn_new_variable = get_combined_dataset_coditional(Yn=Yn) 
            conditional_mi.append(conditionalmutualinformationempirical(xn=xn, yn=Yn, zn=zn_new_variable))

    return {"total_mi": total_mi, "conditional_mi": conditional_mi, "individual_mi": individual_mi}
    

# def multivariate_mutual_information_longitudinal(xn, Yn, xn_index = 0, is_tot_DPL = False):
#     global new_data_list
#     new_data_list = []
#     for i in range(len(Yn)):
#         new_data_list.append("")

#     pair_wise_mi = {}
#     sorted_pair_wise_mi_list = []

#     # tot_DPL = 0

#     for index_i in range(np.shape(Yn)[1]):

#         i = Yn[:,index_i]
#         current_value = mutualinformationempirical(xn, i)[0]
#         pair_wise_mi[index_i] = current_value
#         # if index_i == xn_index:
#         #     tot_DPL = current_value
#         # print(mutualinformationempirical(xn, i)[3], mutualinformationempirical(xn, i)[5])

#         if len(sorted_pair_wise_mi_list) == 0:
#             sorted_pair_wise_mi_list = [index_i]
#         else:
#             is_added = False
#             for index_j, j in enumerate(sorted_pair_wise_mi_list):
#                 if pair_wise_mi[j] < current_value:
#                     sorted_pair_wise_mi_list = sorted_pair_wise_mi_list[:index_j] + [index_i] + sorted_pair_wise_mi_list[index_j:]
#                     is_added = True
#                     break
#             if not(is_added):
#                 sorted_pair_wise_mi_list.append(index_i)
    
#     S = []
#     for i in sorted_pair_wise_mi_list:
#         S.append(pair_wise_mi[i])
#     # print("sorted_pair_wise_mi_list ", sorted_pair_wise_mi_list, "xn_index", xn_index, " S ", S)
    
#     total_mutual_info = 0 # pair_wise_mi[sorted_pair_wise_mi_list[0]]

#     selected_indexes = []
#     conditional_mi_list = []

#     for i in range(xn_index):
#         if i < 1:
#             continue
#         selected_indexes.append(i - 1)
#         zn_new_variable = get_combined_dataset(Yn=Yn, selected_indexes=selected_indexes) # Yn[:,selected_indexes[0]] # get_combined_dataset(Yn=Yn, selected_indexes=selected_indexes)
#         conditional_mi = conditionalmutualinformationempirical(xn=xn, yn=Yn[:,i], zn=zn_new_variable)
#         conditional_mi_list.append((sorted_pair_wise_mi_list[index_i], i, conditional_mi))
#         if conditional_mi < 0.005: # total_mutual_info*0.001:
#             # total_mutual_info += conditional_mi
#             continue

#         total_mutual_info += conditional_mi
#     # print("conditional_mi_list ", conditional_mi_list)
#     # print("total_mutual_info ", total_mutual_info)
#     if is_tot_DPL:
#         return total_mutual_info, pair_wise_mi[xn_index]
#     return total_mutual_info

def total_entropy(encoded_selected_dataset):
    columns = list(encoded_selected_dataset.columns)

    total_entropy = 0

    for index_i, i in enumerate(columns):
        total_entropy += entropyempirical(xn=encoded_selected_dataset.values[:,index_i])[0]
    
    return total_entropy
