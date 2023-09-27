# ## Simulation parameters
exp_name = 'exp-FINAL-1'
eps = 0.10
eps_str = '0_1'

run_idx_length = 1_000
N_val = 1_000
runs = 50

# Number of action-flipped states
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--flip_num', type=int)
parser.add_argument('--flip_seed', type=int)
args = parser.parse_args()
pol_flip_num = args.flip_num
pol_flip_seed = args.flip_seed

pol_name = f'flip{pol_flip_num}_seed{pol_flip_seed}'
out_fname = f'./results/{exp_name}/vaso_eps_{eps_str}-{pol_name}-aug_step-Naive.csv'

import numpy as np
import pandas as pd

df_tmp = None
try:
    df_tmp = pd.read_csv(out_fname)
except:
    pass

if df_tmp is not None:
    print('File exists')
    quit()

from tqdm import tqdm
from collections import defaultdict
import pickle
import itertools
import copy
import random
import itertools
import joblib
from joblib import Parallel, delayed

from OPE_utils_new import (
    format_data_tensor,
    policy_eval_analytic_finite,
    OPE_IS_h,
    compute_behavior_policy_h,
)


def policy_eval_helper(π):
    V_H = policy_eval_analytic_finite(P.transpose((1,0,2)), R, π, gamma, H)
    Q_H = [(R + gamma * P.transpose((1,0,2)) @ V_H[h]) for h in range(1,H)] + [R]
    J = isd @ V_H[0]
    # Check recursive relationships
    assert len(Q_H) == H
    assert len(V_H) == H
    assert np.all(Q_H[-1] == R)
    assert np.all(np.sum(π * Q_H[-1], axis=1) == V_H[-1])
    assert np.all(R + gamma * P.transpose((1,0,2)) @ V_H[-1] == Q_H[-2])
    return V_H, Q_H, J


NSTEPS = H = 20   # max episode length in historical data # Horizon of the MDP
G_min = -1        # the minimum possible return
G_max =  1        # the maximum possible return
nS, nA = 1442, 8

PROB_DIAB = 0.2

# Ground truth MDP model
MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
P = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
R = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)
nS, nA = R.shape
gamma = 0.99

# unif rand isd, mixture of diabetic state
isd = joblib.load('../data/modified_prior_initial_state_absorbing.joblib')
isd = (isd > 0).astype(float)
isd[:720] = isd[:720] / isd[:720].sum() * (1-PROB_DIAB)
isd[720:] = isd[720:] / isd[720:].sum() * (PROB_DIAB)

# Precomputed optimal policy
π_star = joblib.load('../data/π_star.joblib')



# ## Load data
input_dir = f'../datagen/vaso_eps_{eps_str}-100k/'

def load_data(fname):
    print('Loading data', fname, '...', end='')
    df_data = pd.read_csv('{}/{}'.format(input_dir, fname)).rename(columns={'State_idx': 'State'})#[['pt_id', 'Time', 'State', 'Action', 'Reward']]

    # Assign next state
    df_data['NextState'] = [*df_data['State'].iloc[1:].values, -1]
    df_data.loc[df_data.groupby('pt_id')['Time'].idxmax(), 'NextState'] = -1
    df_data.loc[(df_data['Reward'] == -1), 'NextState'] = 1440
    df_data.loc[(df_data['Reward'] == 1), 'NextState'] = 1441

    assert ((df_data['Reward'] != 0) == (df_data['Action'] == -1)).all()

    print('DONE')
    return df_data


# df_seed1 = load_data('1-features.csv') # tr
df_seed2 = load_data('2-features.csv') # va


# ## Naive CF-OPE code

def compute_behavior_policy_weighted(df_data, trajW):
    """
    Calculate probabilities of the behavior policy π_b
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical behavior policy from data
    π_b = np.zeros((nS, nA))
    df_dataW = df_data.set_index('pt_id').join(
        pd.DataFrame(trajW.sum(axis=1), 
                     index=sorted(df_data['pt_id'].unique()), columns=['Weight'])
    ).reset_index()

    sa_counts = df_dataW.groupby(['State', 'Action'])[['Weight']].sum().rename(columns={'Weight': 'count'}).reset_index()
    # df_data.groupby(['State', 'Action']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()

    try:
        for i, row in sa_counts.iterrows():
            s, a = int(row['State']), int(row['Action'])
            count = row['count']
            if row['Action'] == -1:
                π_b[s, :] = count
            else:
                π_b[s, a] = count
    except:
        print(s,a)
        raise
        # import pdb
        # pdb.set_trace()

    # assume uniform action probabilities in unobserved states
    unobserved_states = (π_b.sum(axis=-1) == 0)
    π_b[unobserved_states, :] = 1

    # normalize action probabilities
    π_b = π_b / π_b.sum(axis=-1, keepdims=True)

    return π_b


def OPE_IS_trajW(data, π_b, π_e, γ, epsilon=0.01, trajW=None):
    """
    - π_b, π_e: behavior/evaluation policy, shape (S,A)
    """
    # Get a soft version of the evaluation policy for WIS
    π_e_soft = np.copy(π_e).astype(float)
    π_e_soft[π_e_soft == 1] = (1 - epsilon)
    π_e_soft[π_e_soft == 0] = epsilon / (nA - 1)
    
    # Apply WIS
    return _is_trajW(data, π_b, π_e_soft, γ, np.eye(len(data)) if trajW is None else trajW)

def _is_trajW(data, π_b, π_e, γ, trajW):
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
    p_b = π_b[s_list, a_list]
    p_e = π_e[s_list, a_list]

    # Deal with variable length sequences by setting ratio to 1
    terminated_idx = (a_list == -1)
    p_b[terminated_idx] = 1
    p_e[terminated_idx] = 1
    
    # if not np.all(p_b > 0):
    #     import pdb
    #     pdb.set_trace()
    # assert np.all(p_b > 0), "Some actions had zero prob under p_b, WIS fails"

    # Per-trajectory cumulative importance ratios, take the product
    rho = (p_e / p_b).prod(axis=1) * trajW.sum(axis=1)

    # directly calculate weighted average over trajectories
    is_value = np.nansum(G * rho) / trajW.shape[1]
    wis_value = np.nansum(G * rho) / np.nansum(rho)
    rho_norm = rho / np.nansum(rho)
    rho_nna = rho[~np.isnan(rho)]
    rho_norm_nna = rho_norm[~np.isnan(rho_norm)]
    ess1 = 1 / np.nansum(rho_norm_nna ** 2)
    ess1_ = (np.nansum(rho_nna)) ** 2 / (np.nansum(rho_nna ** 2))
    # assert np.isclose(ess1, ess1_)
    ess2 = 1. / np.nanmax(rho_norm)
    return is_value, wis_value, {
        'ESS1': ess1, 'ESS2': ess2, 'G': G,
        'rho': rho, 'rho_norm': rho_norm
    }


# ## Policies

# vaso unif, mv abx optimal
π_unif = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)


# ### Behavior policy

# vaso eps=0.5, mv abx optimal
π_beh = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)
π_beh[π_star == 1] = 1-eps
π_beh[π_beh == 0.5] = eps

V_H_beh, Q_H_beh, J_beh = policy_eval_helper(π_beh)
J_beh


# ### Optimal policy
V_H_star, Q_H_star, J_star = policy_eval_helper(π_star)
J_star


# ### flip action for x% states

rng_flip = np.random.default_rng(pol_flip_seed)
flip_states = rng_flip.choice(range(1440), pol_flip_num, replace=False)

π_tmp = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)
π_flip = π_tmp.copy()
π_flip[π_tmp == 0.5] = 0
π_flip[π_star == 1] = 1
for s in flip_states:
    π_flip[s, π_tmp[s] == 0.5] = 1
    π_flip[s, π_star[s] == 1] = 0
assert π_flip.sum(axis=1).mean() == 1

# np.savetxt(f'./results/{exp_name}/policy_{pol_name}.txt', π_flip)


# ## Compare OPE

π_eval = π_flip


# ### Proposed: replace future with the value function for the evaluation policy

df_va_all2 = pd.read_pickle(f'results/vaso_eps_{eps_str}-annotOpt_df_seed2_aug_step.pkl')
V_H_eval, Q_H_eval, J_eval = policy_eval_helper(π_eval)


df_results_v2 = []
for run in range(runs):
    df_va = df_va_all2.set_index('pt_id').loc[200000+run*run_idx_length:200000+run*run_idx_length + N_val - 1+0.999].reset_index()
    
    # m is num of trajectories
    # (m_all, m_orig) table with binary indicator of the source (original traj) of each traj
    df_idmap = df_va[['pt_id']].copy()
    df_idmap['map_pt_id'] = df_va['pt_id'].apply(np.floor).astype(int)
    df_idmap['mask'] = 1
    df_mapp = df_idmap.drop_duplicates().set_index(['pt_id', 'map_pt_id']).unstack().fillna(0).astype(pd.SparseDtype('int', 0))

    # (m_all, m_orig) traj-wise weight matrix
    # each column sums to 1
    traj_weight_matrix = df_mapp.values
    traj_weight_matrix = (traj_weight_matrix / traj_weight_matrix.sum(axis=0))
    assert np.isclose(traj_weight_matrix.sum(axis=0), 1).all()

    # OPE - WIS/WDR prep
    v2_data_va = format_data_tensor(df_va)
    v2_pi_b_val = compute_behavior_policy_weighted(df_va, traj_weight_matrix)

    # OPE - WIS
    v2_IS_value, v2_WIS_value, v2_ESS_info = OPE_IS_trajW(v2_data_va, v2_pi_b_val, π_eval, gamma, trajW=traj_weight_matrix)

    df_results_v2.append([v2_IS_value, v2_WIS_value, v2_ESS_info['ESS1'], v2_ESS_info['ESS2']])
    pd.DataFrame(df_results_v2, columns=['IS_value', 'WIS_value', 'ESS1', 'ESS2']).to_csv(out_fname)

df_results_v2 = pd.DataFrame(df_results_v2, columns=['IS_value', 'WIS_value', 'ESS1', 'ESS2'])
df_results_v2.to_csv(out_fname)
