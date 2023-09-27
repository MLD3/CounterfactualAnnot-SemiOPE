import numpy as np
import pandas as pd
import numpy_indexed as npi
import joblib
from tqdm import tqdm
import itertools
import copy

NSTEPS = H = 20       # max episode length in historical data
G_min = -1        # the minimum possible return
G_max =  1        # the maximum possible return
nS, nA = 1442, 8

##################
## Preparations ##
##################

def format_data_tensor(df_data, id_col='pt_id'):
    """
    Converts data from a dataframe to a tensor
    - df_data: pd.DataFrame with columns [id_col, Time, State, Action, Reward, NextState]
        - id_col specifies the index column to group episodes
    - data_tensor: integer tensor of shape (N, NSTEPS, 5) with the last last dimension being [t, s, a, r, s']
    """
    data_dict = dict(list(df_data.groupby(id_col)))
    N = len(data_dict)
    data_tensor = np.zeros((N, NSTEPS, 5), dtype=float)
    data_tensor[:, :, 2] = -1 # initialize all actions to -1
    data_tensor[:, :, 1] = -1 # initialize all states to -1
    data_tensor[:, :, 4] = -1 # initialize all next states to -1

    for i, (pt_id, df_values) in tqdm(enumerate(data_dict.items()), disable=True):
        values = df_values.set_index(id_col).values
        data_tensor[i, :len(values), :] = values
    return data_tensor

def compute_behavior_policy(df_data):
    """
    Calculate probabilities of the behavior policy π_b
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical behavior policy from data
    π_b = np.zeros((nS, nA))
    sa_counts = df_data.groupby(['State', 'Action']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()

    for i, row in sa_counts.iterrows():
        s, a = row['State'], row['Action']
        count = row['count']
        if row['Action'] == -1:
            π_b[s, :] = count
        else:
            π_b[s, a] = count

    # assume uniform action probabilities in unobserved states
    unobserved_states = (π_b.sum(axis=-1) == 0)
    π_b[unobserved_states, :] = 1

    # normalize action probabilities
    π_b = π_b / π_b.sum(axis=-1, keepdims=True)

    return π_b

def compute_behavior_policy_h(df_data):
    """
    Calculate probabilities of the behavior policy π_b
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical behavior policy from data
    πh_b = np.zeros((H, nS, nA))
    hsa_counts = df_data.groupby(['Time', 'State', 'Action']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()

    for i, row in hsa_counts.iterrows():
        h, s, a = row['Time'], row['State'], row['Action']
        count = row['count']
        if row['Action'] == -1:
            πh_b[h, s, :] = count
        else:
            πh_b[h, s, a] = count

    # assume uniform action probabilities in unobserved states
    unobserved_states = (πh_b.sum(axis=-1) == 0)
    πh_b[unobserved_states, :] = 1

    # normalize action probabilities
    πh_b = πh_b / πh_b.sum(axis=-1, keepdims=True)

    return πh_b

#########################
## Evaluating a policy ##
#########################

def policy_eval_analytic(P, R, π, γ):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, nA = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = np.linalg.inv(np.eye(nS) - γ * P_π) @ R_π
    return V_π

def policy_eval_analytic_finite(P, R, π, γ, H):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using the power series formula
    Horizon h=1...H
        V_π(h) = R_π + γ P_π R_π + ... + γ^{h-1} P_π^{h-1} R_π
    """
    nS, nA = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = [R_π]
    for h in range(1,H):
        V_π.append(R_π + γ * P_π @ V_π[-1])
    return list(reversed(V_π))

def OPE_IS_h(data, π_b, π_e, γ, epsilon=0.01):
    """
    - π_b, π_e: behavior/evaluation policy, shape (S,A)
    """
    # Get a soft version of the evaluation policy for WIS
    π_e_soft = np.copy(π_e).astype(float)
    π_e_soft[π_e_soft == 1] = (1 - epsilon)
    π_e_soft[π_e_soft == 0] = epsilon / (nA - 1)
    
    # Apply WIS
    return _is_h(data, π_b, π_e_soft, γ)

def _is_h(data, π_b, π_e, γ):
    """
    Weighted Importance Sampling for Off-Policy Evaluation
        - data: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        - π_b:  behavior policy
        - π_e:  evaluation policy (aka target policy)
        - γ:    discount factor
    """
    t_list = data[..., 0].astype(int)
    s_list = data[..., 1].astype(int)
    a_list = data[..., 2].astype(int)
    r_list = data[..., 3].astype(float)
    
    # Per-trajectory returns (discounted cumulative rewards)
    G = (r_list * np.power(γ, t_list)).sum(axis=-1)
    
    # Per-transition importance ratios
    p_b = π_b[t_list, s_list, a_list]
    p_e = π_e[s_list, a_list]

    # Deal with variable length sequences by setting ratio to 1
    terminated_idx = (a_list == -1)
    p_b[terminated_idx] = 1
    p_e[terminated_idx] = 1
    
    if not np.all(p_b > 0):
        import pdb
        pdb.set_trace()
    assert np.all(p_b > 0), "Some actions had zero prob under p_b, WIS fails"

    # Per-trajectory cumulative importance ratios, take the product
    rho = (p_e / p_b).prod(axis=1)
    rho_norm = rho / rho.sum()

    # directly calculate weighted average over trajectories
    is_value = np.average(G*rho) # (G @ rho) / len(G)
    wis_value = np.average(G, weights=rho) # (G @ rho_norm)
    ess1 = 1 / (rho_norm ** 2).sum()
    ess1_ = (rho.sum()) ** 2 / ((rho ** 2).sum())
    assert np.isclose(ess1, ess1_)
    ess2 = 1. / rho_norm.max()
    return is_value, wis_value, {
        'ESS1': ess1, 'ESS2': ess2, 'G': G,
        'rho': rho, 'rho_norm': rho_norm
    }
