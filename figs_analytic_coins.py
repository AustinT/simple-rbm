import numpy as np
import rbm
import os
from scipy import stats
import pickle as pkl
import random
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--load", action="store_true", help="Use to load values instead of doing the simulation")
parser.add_argument("--plot", action="store_true", help="Whether to make plots.")
parser.add_argument("--show", action="store_true", help="Whether to show plots.")
parser.add_argument("--n-epochs", type=int, default=100, help="Number of Epochs")
parser.add_argument("--epoch-length", type=int, default=5000, help="Length of each epoch")

# Unique arguments for this
parser.add_argument("-d", type=int, default=0, help="Which distribution to use")

# Parse args
args = parser.parse_args()
DIST_NUM = args.d

# The different distributions to use
distributions = [{(1.0,1.0): 0.25, (1.0,0.0): 0.25, (0.0, 1.0): 0.25, (0.0,0.0): 0.25},
                 {(1.0,1.0): 0.49, (1.0,0.0): 0.01, (0.0, 1.0): 0.01, (0.0,0.0): 0.49},
                 {(1.0,1.0): 0.1, (1.0,0.0): 0.2, (0.0, 1.0): 0.45, (0.0,0.0): 0.25},
                 ]

assert all([np.isclose(sum(d.values()), 1) for d in distributions])

# Fixed parameters
N_coins = 2
n_v = N_coins
n_h = 2
learning_rate = 2e-3
weight_decay = 0

# Fix the random seeds
np.random.seed(543)
random.seed(7607)

# Coin stats Th
def get_stats(state_list):
    
    overall_avg = np.average(state_list)
    per_coin_avg = np.average(state_list, axis=0)
    cov = np.cov(state_list, rowvar=False)
    return dict(cov=cov, avg=overall_avg, avg1=per_coin_avg[0], avg2=per_coin_avg[1])

# Method to run simulations
def run_simulations():

    # Do "Monte Carlo" over the analytic distribution
    print("Doing MCMC")
    probs_dict = distributions[DIST_NUM]
    mcmc_binary_states_list = []
    states_list = []
    probs_list = []
    for state, prob in probs_dict.items():
        states_list.append(np.array(state))
        probs_list.append(prob)
    probs_list = np.array(probs_list)
    for _ in tqdm(range(int(1e5))):
        i = np.random.choice(len(probs_list), p=probs_list)
        mcmc_binary_states_list.append(states_list[i].copy())
    
    # Create 3 models for 3 different training regimes: exact GD, SGD, and CD
    # Each model starts with the same initialization
    w_start = np.random.normal(loc=0, scale=1., size=(n_h, n_v))
    model_exact = rbm.RBM(n_v, n_h, w=w_start.copy())
    model_stochastic = rbm.RBM(n_v, n_h, w=w_start.copy())
    model_cd = rbm.RBM(n_v, n_h, w=w_start.copy())
    
    # Function to evaluate KL divergence given the exact probability
    def eval_kl(model):
        model._get_z()
        p_true_list = []
        p_rbm_list = []
        for state, p_true in probs_dict.items():
            p_rbm = model._get_marginalized_prob(v=np.array(state))
            p_rbm_list.append(p_rbm)
            p_true_list.append(p_true)
        kl_div = stats.entropy(p_rbm_list, p_true_list)
        return kl_div
        
    # Track KL divergence analytically during the training process
    kl_exact = []
    kl_stochastic = []
    kl_cd = []
    def track_kls():
        kl_exact.append(eval_kl(model_exact))
        kl_stochastic.append(eval_kl(model_stochastic))
        kl_cd.append(eval_kl(model_cd))
    # Track the KL divergence before training starts
    track_kls()

    # Track statistics about the distributions
    stats_exact = []
    stats_stochastic = []
    stats_cd = []
    def track_stats(model, stats_list, n_sample=int(1e4)):
        states = model.gibbs_sample_array(n_sample, k=3)
        stats = get_stats(states)
        stats_list.append(stats)
        return stats

    print("\nStarting training...")
    for epoch in tqdm(range(args.n_epochs)):
    
        # Shuffle training data
        random.shuffle(mcmc_binary_states_list)

        for t in tqdm(range(args.epoch_length)):
            grads_exact = model_exact._get_exact_gradients(probs_dict)
            model_exact.apply_gradients(grads_exact, learning_rate=learning_rate)
            
            grads_stochastic = model_stochastic._get_exact_single_sample_gradients(mcmc_binary_states_list[t])
            model_stochastic.apply_gradients(grads_stochastic, learning_rate=learning_rate)
            
            grads_cd = model_cd.get_contrastive_divergence(mcmc_binary_states_list[t], k=3)
            model_cd.apply_gradients(grads_cd, learning_rate=learning_rate)

            
        # Track the KL div
        track_kls()

        # Track stats
        track_stats(model_exact, stats_exact)
        track_stats(model_stochastic, stats_stochastic)
        track_stats(model_cd, stats_cd)
       
    # Turn KL divergence into arrays
    kl_exact = np.array(kl_exact)
    kl_stochastic = np.array(kl_stochastic)
    kl_cd = np.array(kl_cd)

    # Turn stats into arrays
    def stats_list_to_array(stats_list):
        d = stats_list[0]
        d_out = dict()
        for key in d:
            d_out[key] = np.array([x[key] for x in stats_list])
        return d_out
    stats_exact = stats_list_to_array(stats_exact)
    stats_stochastic = stats_list_to_array(stats_stochastic)
    stats_cd = stats_list_to_array(stats_cd)
    stats_mcmc = get_stats(mcmc_binary_states_list)
    
    # Save results to dictionary
    results_dict = dict(kl_exact=kl_exact, kl_stochastic=kl_stochastic, kl_cd=kl_cd,
                        stats_exact=stats_exact, stats_stochastic=stats_stochastic, stats_cd=stats_cd,
                        stats_mcmc=stats_mcmc,
                        model_cd=model_cd, model_stochastic=model_stochastic, model_exact=model_exact)
    return results_dict

# Saving/loading of data
all_results = dict()
save_file_name = "coins_analytic_{}.pkl".format(DIST_NUM)
if args.load:
    print("LOADING MODEL...")
    with open(save_file_name, "rb") as f:
        all_results = pkl.load(f)
else:
    all_results = run_simulations()
    with open(save_file_name, "wb") as f:
        pkl.dump(all_results, f)

if args.plot:
    import matplotlib.pyplot as plt

    # What the different legend items are called
    legend_map = dict(_exact="GD", _stochastic="SGD", _cd="CD")
    num_x_points = len(all_results["kl_cd"])

    # Plot KL divergence vs time for single distribution
    for key, legend_name  in sorted(legend_map.items()):
        plt.semilogy(all_results["kl"+key], '.-', label=legend_name)
    plt.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("KL-Divergence")
    plt.title("KL-Divergence vs Iteration Number for 2 Coin Distribution {}\nUsing Various RBM Training Techniques".format("ABCDEFG"[DIST_NUM]))
    plt.tight_layout()
    plt.savefig("coins_kldiv_{}.png".format(DIST_NUM))
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot the evolution of the different stats
    # Two coin averages
    for key, legend_name  in sorted(legend_map.items()):
        stats_dict = all_results["stats"+key]
        plt.plot(stats_dict["avg"], '.-', label=legend_name)
    plt.plot([all_results["stats_mcmc"]["avg"]]*num_x_points, '--', label="MCMC") 
    plt.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("Average Value")
    plt.title("Average Value vs Iteration Number for 2 Coin Distribution {}\nUsing Various RBM Training Techniques".format("ABCDEFG"[DIST_NUM]))
    plt.tight_layout()
    plt.savefig("coins_avg_{}.png".format(DIST_NUM))
    if args.show:
        plt.show()
    else:
        plt.close()

    # Variance of coin 1
    for key, legend_name  in sorted(legend_map.items()):
        stats_dict = all_results["stats"+key]
        plt.plot(stats_dict["cov"][:, 0, 0], '.-', label=legend_name)
    plt.plot([all_results["stats_mcmc"]["cov"][0, 0]]*num_x_points, '--', label="MCMC") 
    plt.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("Variance of Coin 1")
    plt.title("Variance of Coin 1 vs Iteration Number for 2 Coin Distribution {}\nUsing Various RBM Training Techniques".format("ABCDEFG"[DIST_NUM]))
    plt.tight_layout()
    plt.savefig("coin1_var_{}.png".format(DIST_NUM))
    if args.show:
        plt.show()
    else:
        plt.close()   

    # Coin covariances
    for key, legend_name  in sorted(legend_map.items()):
        stats_dict = all_results["stats"+key]
        plt.plot(stats_dict["cov"][:, 0, 1], '.-', label=legend_name)
    plt.plot([all_results["stats_mcmc"]["cov"][0, 1]]*num_x_points, '--', label="MCMC") 
    plt.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("Covariance of 2 coins")
    plt.title("Covariance of Coins vs Iteration Number for 2 Coin Distribution {}\nUsing Various RBM Training Techniques".format("ABCDEFG"[DIST_NUM]))
    plt.tight_layout()
    plt.savefig("coins_covar_{}.png".format(DIST_NUM))
    if args.show:
        plt.show()
    else:
        plt.close()   

