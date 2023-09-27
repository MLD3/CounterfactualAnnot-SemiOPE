# ## Simulation parameters
exp_name = 'exp-FINAL-5'
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
out_fname = f'./results/{exp_name}/vaso_eps_{eps_str}-{pol_name}-aug_step-annotBehConvertedAM.csv'

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

from OPE_utils import (
    compute_behavior_policy,
    compute_empirical_MDP,
)
from OPE_utils_new import (
    format_data_tensor,
    policy_eval_analytic_finite,
    OPE_IS_h,
    compute_behavior_policy_h,
)


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


def policy_eval_helper(π, P=P, R=R, isd=isd):
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

def iqm(x):
    return scipy.stats.trim_mean(x, proportiontocut=0.25, axis=None)


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


# ## C-PDIS code

def compute_augmented_behavior_policy_h(df_data):
    πh_b = np.zeros((H, nS, nA))
    hsa_counts = df_data.groupby(['Time', 'State', 'Action'])[['Weight']].sum().rename(columns={'Weight': 'count'}).reset_index()

    try:
        for i, row in hsa_counts.iterrows():
            h, s, a = int(row['Time']), int(row['State']), int(row['Action'])
            count = row['count']
            if row['Action'] == -1:
                πh_b[h, s, :] = count
            else:
                πh_b[h, s, a] = count
    except:
        print(h,s,a)
        raise
        # import pdb
        # pdb.set_trace()

    # assume uniform action probabilities in unobserved states
    unobserved_states = (πh_b.sum(axis=-1) == 0)
    πh_b[unobserved_states, :] = 1

    # normalize action probabilities
    πh_b = πh_b / πh_b.sum(axis=-1, keepdims=True)

    return πh_b


def format_data_tensor_cf(df_data, id_col='map_pt_id'):
    """
    Converts data from a dataframe to a tensor
    - df_data: pd.DataFrame with columns [id_col, Time, State, Action, Reward, NextState]
        - id_col specifies the index column to group episodes
    - data_tensor: integer tensor of shape (N, NSTEPS, 5) with the last last dimension being [t, s, a, r, s']
    """
    data_dict = dict(list(df_data.groupby(id_col)))
    N = len(data_dict)
    data_tensor = np.zeros((N, 2*NSTEPS, 6), dtype=float)
    data_tensor[:, :, 0] = -1 # initialize all time steps to -1
    data_tensor[:, :, 2] = -1 # initialize all actions to -1
    data_tensor[:, :, 1] = -1 # initialize all states to -1
    data_tensor[:, :, 4] = -1 # initialize all next states to -1
    data_tensor[:, :, 5] = np.nan # initialize all weights to NaN

    for i, (pt_id, df_values) in tqdm(enumerate(data_dict.items()), disable=True):
        values = df_values.set_index(id_col)[['Time', 'State', 'Action', 'Reward', 'NextState', 'Weight']].values
        data_tensor[i, :len(values), :] = values
    return data_tensor


def OPE_PDIS_h(data, π_b, π_e, γ, epsilon=0.01):
    """
    - π_b, π_e: behavior/evaluation policy, shape (S,A)
    """
    # Get a soft version of the evaluation policy for WIS
    π_e_soft = π_e.astype(float) * (1 - epsilon*2) + π_unif * epsilon*2
    
    # # Get a soft version of the behavior policy for WIS
    # π_b_soft = π_b * (1 - epsilon) + epsilon / nA
    
    # Apply WIS
    return _pdis_h(data, π_b, π_e_soft, γ)

def _pdis_h(data, π_b, π_e, γ):
    # For each original trajectory
    v_all, rho_all = [], []
    for i, data_i in enumerate(data):
        # Get all trajectories based on this trajectory
        t_l = data_i[..., 0].astype(int)
        s_l = data_i[..., 1].astype(int)
        a_l = data_i[..., 2].astype(int)
        r_l = data_i[..., 3].astype(float)
        snext_l = data_i[..., 4].astype(int)
        w_l = data_i[..., 5].astype(float)

        # Per-transition importance ratios
        p_b = π_b[t_l, s_l, a_l]
        p_e = π_e[s_l, a_l]

        # Deal with variable length sequences by setting ratio to 1
        terminated_idx = (a_l == -1)
        terminating_idx = (s_l != -1) & (a_l == -1)
        p_b[terminated_idx] = np.nan
        p_e[terminated_idx] = np.nan
        p_b[terminating_idx] = 1
        p_e[terminating_idx] = 1

        # Per-step cumulative importance ratios
        rho_t = (p_e / p_b)

        # # Last observed step of each trajectory
        # idx_last = np.array([np.max(np.nonzero(s_l[row] != -1)) for row in range(len(s_l))])

        # Initialize value to 0, importance ratio to 1
        v = 0
        rho_cum = 1

        # Iterate backwards from step H to 1
        for h in reversed(range(H)):
            # only start computing from the last observed step
            if not (t_l == h).any():
                continue

            # do we have counterfactual annotation for this step?
            if (t_l == h).sum() > 1:
                # if we have counterfactual annotations for this step
                j_all = np.argwhere(t_l == h).ravel()
                assert np.isclose(w_l[j_all].sum(), 1) # weights add up to 1

                # Identify factual transition and counterfactual annotations
                f_, cf_ = [], []
                for j in j_all:
                    if snext_l[j] == 1442:  # counterfactual annotation have dummy next state
                        cf_.append(j)
                    else:
                        f_.append(j)
                assert len(f_) == 1 # there should only be one factual transition
                f_ = f_[0]
                v = w_l[f_]*rho_t[f_]*(r_l[f_]+γ*v) + np.sum([w_l[j]*rho_t[j]*r_l[j] for j in cf_])
                rho_cum = rho_cum * (w_l[f_]*rho_t[f_]) + np.sum([w_l[j]*rho_t[j] for j in cf_])
            else:
                # we don't have counterfactual annotations for this step
                # there should only be one trajectory and that must be the original traj
                j = (t_l == h).argmax()
                assert ~np.isnan(p_e[j])
                assert w_l[j] == 1.0
                v = rho_t[j] * (r_l[j]+γ*v)
                rho_cum = rho_cum * rho_t[j]

        v_all.append(v)
        rho_all.append(rho_cum)

    v_all = np.array(v_all)
    rho_all = np.array(rho_all)
    is_value = np.nansum(v_all) / len(rho_all)
    wis_value = np.nansum(v_all) / np.nansum(rho_all)
    rho_norm = rho_all / np.nansum(rho_all)
    rho_nna = rho_all[~np.isnan(rho_all)]
    rho_norm_nna = rho_norm[~np.isnan(rho_norm)]
    ess1 = 1 / np.nansum(rho_norm_nna ** 2)
    ess1_ = (np.nansum(rho_nna)) ** 2 / (np.nansum(rho_nna ** 2))
    ess2 = 1. / np.nanmax(rho_norm)
    return is_value, wis_value, {
        'ESS1': ess1, 'ESS2': ess2, 
        'rho': rho_all, 'rho_norm': rho_norm_nna,
    }


## Default weighting scheme for C-PDIS

weight_a_sa = np.zeros((nS, nA, nA))

# default weight if no counterfactual actions
for a in range(nA):
    weight_a_sa[:, a, a] = 1

# split equally between factual and counterfactual actions
for s in range(nS):
    a = π_star.argmax(axis=1)[s]
    a_tilde = a+1-2*(a%2)
    weight_a_sa[s, a, a] = 0.5
    weight_a_sa[s, a, a_tilde] = 0.5
    weight_a_sa[s, a_tilde, a] = 0.5
    weight_a_sa[s, a_tilde, a_tilde] = 0.5

assert np.all(weight_a_sa.sum(axis=-1) == 1)


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
    
    # Original dataset
    df_va_orig = df_seed2.set_index('pt_id').loc[200000+run*run_idx_length:200000+run*run_idx_length + N_val - 1].reset_index()[
        ['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']
    ]
    df_orig = df_va_orig[['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']]
    data_orig = format_data_tensor(df_orig)
    pi_b_orig = compute_behavior_policy(df_orig)
    
    # Build approx MDP and compute Q-functions for both π_b and π_e
    P_approx, R_approx, isd_approx = compute_empirical_MDP(df_va_orig)
    V_H_eval_approx, Q_H_eval_approx, J_eval_approx = policy_eval_helper(π_eval, P=P_approx, R=R_approx, isd=isd_approx)
    V_H_beh_approx, Q_H_beh_approx, J_beh_approx = policy_eval_helper(pi_b_orig, P=P_approx, R=R_approx, isd=isd_approx)
    
    # Augmented dataset
    df_va = df_va_all2.set_index('pt_id').loc[200000+run*run_idx_length:200000+run*run_idx_length + N_val - 1+0.999].reset_index()
    df_va['map_pt_id'] = df_va['pt_id'].apply(np.floor).astype(int)
    df_va = df_va.drop_duplicates(['map_pt_id', 'Time', 'State', 'Action']) \
        .sort_values(by=['map_pt_id', 'Time', 'pt_id']).reset_index(drop=True)
    df_va['Weight'] = np.nan
    
    for i, row in tqdm(list(df_va.iterrows()), disable=True):
        map_pt_id = row['map_pt_id']
        h = int(row['Time'])
        s = int(row['State'])
        if row['NextState'] in [1440, 1441]:
            df_va.loc[i, 'Weight'] = 1.0
        elif row['NextState'] in [1442]:
            a_cf = int(row['Action'])
            a_f = int(a_cf+1-2*(a_cf%2))
            df_va.loc[i, 'Weight'] = weight_a_sa[s, a_f, a_cf]
            df_va.loc[(df_va['map_pt_id'] == map_pt_id) 
                      & (df_va['Time'] == h) 
                      & (df_va['Action'] == a_f), 'Weight'] = weight_a_sa[s, a_f, a_f]
            annot_cf = Q_H_beh[h][s,a_cf] - Q_H_beh_approx[h][s,a_cf] + Q_H_eval_approx[h][s,a_cf]
            df_va.loc[i, 'Reward'] = annot_cf
        else:
            pass
    df_va['Weight'] = df_va['Weight'].fillna(1)
    
    # OPE - WIS/WDR prep
    v2_data_va = format_data_tensor_cf(df_va)
    v2_pi_b_val = compute_augmented_behavior_policy_h(df_va)

    # OPE - WIS
    v2_IS_value, v2_WIS_value, v2_ESS_info = OPE_PDIS_h(v2_data_va, v2_pi_b_val, π_eval, gamma, epsilon=0.0)

    df_results_v2.append([v2_IS_value, v2_WIS_value, v2_ESS_info['ESS1']])
    pd.DataFrame(df_results_v2, columns=['IS_value', 'WIS_value', 'ESS1']).to_csv(out_fname, index=False)

df_results_v2 = pd.DataFrame(df_results_v2, columns=['IS_value', 'WIS_value', 'ESS1'])
df_results_v2.to_csv(out_fname, index=False)
