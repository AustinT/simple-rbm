import ising
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
parser.add_argument("--n-epochs", type=int, default=25, help="Number of Epochs")
parser.add_argument("--epoch-length", type=int, default=10000, help="Length of each epoch")
parser.add_argument("--n-gibbs", type=int, default=10000, help="Number of Gibbs samples to draw")

# Specific to this
parser.add_argument("--nmc", type=int, default=int(1e6), help="Number of Monte Carlo steps")
parser.add_argument("-L", type=int, default=2, help="Length of Ising Square")
parser.add_argument("--fine-tune", action="store_true", help="Whether to train from scratch, or fine tune the model")
parser.add_argument("--beta", nargs="*", type=float, default=[1.])

# Parse args
args = parser.parse_args()

# Fixed parameters
L = args.L
n_v = L**2
n_h = 2*L**2
learning_rate = 1e-2
weight_decay = 0
beta_list = list(args.beta)

# Identifier of this run
identifier = "L={}".format(L)
if args.fine_tune:
    identifier += "_ft"

# Fix the random seeds
np.random.seed(543)
random.seed(7607)

# Stats relevant to Ising model
def get_stats(state_list):

    # Convert binary states to plus minus 1
    pm1_states = ising.debinarize_states_array(state_list)
    
    mag = np.average(pm1_states, axis=1)
    
    # Calculate unit energies
    energies = get_ising_energies(pm1_states) / L**2
    
    return dict(avg_mag=np.average(mag), mag_var=np.var(mag), avg_energy=np.average(energies),
                energy_var=np.var(energies))
    
def get_ising_energies(pm1_states):
    assert pm1_states.shape[1] == L**2
    padded_states = np.zeros((pm1_states.shape[0], L+2, L+2))
    padded_states[:, 1:-1, 1:-1] = pm1_states.reshape(-1, L, L)
    energies = np.array([ising.ising_total_energy(s) for s in padded_states])
    return energies

# Method to run simulations
def run_simulations():
    # Simulate over all beta values
    for beta in beta_list:
        print("STARTING beta = {}".format(beta))
        
        # Do Monte Carlo simulations at this temperature
        print("Doing MCMC")
        res = ising.ising_monte_carlo(int(1e5), args.nmc, L, beta, store_results_every=10)
        mcmc_states = res["states"]

        # Convert the Ising States into binary
        mcmc_binary_states = ising.binarize_states_array(mcmc_states)
        mcmc_binary_states_flat = mcmc_binary_states.reshape(mcmc_binary_states.shape[0], -1)
        mcmc_binary_states_list = list(mcmc_binary_states_flat)
        
        # Initialize the model
        w_start = np.random.normal(loc=0, scale=0.1, size=(n_h, n_v))
        model = rbm.RBM(n_v, n_h, w=w_start.copy())
        
        # Keep track of statistics
        stats_list = []
        def track_stats(model, stats_list, n_sample=int(1e4)):
            states = model.gibbs_sample_array(n_sample, k=3)
            stats = get_stats(states)
            stats_list.append(stats)
            return stats
        track_stats(model, stats_list) # Run once at the start of the model
        
        print("\nStarting training...")
        for epoch in tqdm(range(args.n_epochs)):
        
            # Shuffle training data
            random.shuffle(mcmc_binary_states_list)

            for t in tqdm(range(args.epoch_length)):             
                grads_cd = model.get_contrastive_divergence(mcmc_binary_states_list[t], k=10)
                model.apply_gradients(grads_cd, learning_rate=learning_rate)

            # Track stats
            track_stats(model, stats_list)

        # Turn stats into arrays
        def stats_list_to_array(stats_list):
            d = stats_list[0]
            d_out = dict()
            for key in d:
                d_out[key] = np.array([x[key] for x in stats_list])
            return d_out
        stats_list = stats_list_to_array(stats_list)
        stats_mcmc = get_stats(mcmc_binary_states_flat)
        
        # Save results to dictionary
        results_dict = dict(model=model, beta=beta,L=L, stats_list=stats_list, stats_mcmc=stats_mcmc)
        all_results[beta] = results_dict
    
    return all_results

# Saving/loading of data
all_results = dict()
save_file_name = "ising_analytic_{}.pkl".format(identifier)
if args.load:
    with open(save_file_name, "rb") as f:
        all_results = pkl.load(f)
else:
    all_results = run_simulations()
    with open(save_file_name, "wb") as f:
        pkl.dump(all_results, f)

if args.plot:
    import matplotlib.pyplot as plt
    
    for beta_sample in sorted(list(all_results.keys())):
        num_x_points = len(all_results[beta_sample]["stats_list"]["avg_mag"])

        # Plot magnetization stats
        plt.plot(all_results[beta_sample]["stats_list"]["avg_mag"], '.-', label=r"$\langle M \rangle_{RBM}$")
        plt.plot([all_results[beta_sample]["stats_mcmc"]["avg_mag"]]*num_x_points, '.--', label=r"$\langle M \rangle_{MCMC}$")
        plt.plot(all_results[beta_sample]["stats_list"]["mag_var"], 'x-', label=r"$(\langle M^2 \rangle - \langle M \rangle^2)_{RBM}$")
        plt.plot([all_results[beta_sample]["stats_mcmc"]["mag_var"]]*num_x_points, 'x--', label=r"$(\langle M^2 \rangle - \langle M \rangle^2)_{MCMC}$")
        plt.legend()
        plt.xlabel("Iteration Number")
        plt.ylabel("Magnetization")
        plt.title("Magnetization Average and Variance for RBM Samples During Training\n"+
                  r"$\beta={}$, L={}".format(beta_sample, L))
        plt.tight_layout()
        plt.savefig("figs/ising_mag_{}_beta={}.png".format(identifier, beta_sample))
        if args.show:
            plt.show()
        else:
            plt.close()
        
        # Plot energy stats
        plt.plot(all_results[beta_sample]["stats_list"]["avg_energy"], '.-', label=r"$\langle E \rangle_{RBM}$")
        plt.plot([all_results[beta_sample]["stats_mcmc"]["avg_energy"]]*num_x_points, '.--', label=r"$\langle E \rangle_{MCMC}$")
        plt.plot(all_results[beta_sample]["stats_list"]["energy_var"], 'x-', label=r"$\left(\langle E^2 \rangle - \langle E \rangle^2\right)_{RBM}$")
        plt.plot([all_results[beta_sample]["stats_mcmc"]["energy_var"]]*num_x_points, 'x--', label=r"$\left(\langle E^2 \rangle - \langle E \rangle^2\right)_{MCMC}$")
        plt.legend()
        plt.xlabel("Iteration Number")
        plt.ylabel("Energy")
        plt.title("Energy Average and Variance for RBM Samples During Training\n"+
                  r"$\beta={}$, L={}".format(beta_sample, L))
        plt.tight_layout()
        plt.savefig("figs/ising_energy_{}_beta={}.png".format(identifier, beta_sample))
        if args.show:
            plt.show()
        else:
            plt.close()

