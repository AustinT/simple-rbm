import numpy as np
import math
from tqdm import tqdm
import random

def ising_total_energy(grid):
    l = len(grid) - 2
    energy = 0.
    for i in range(l):
        x = i+1
        for j in range(l):
            y = j+1
            energy += (grid[x-1][y] + grid[x][y-1] + 
                       grid[x+1][y] + grid[x][y+1]) * grid[x][y]
    return -energy / 2.

def get_grid_sum(grid):
    return sum(sum(l) for l in grid)

def ising_monte_carlo(num_eq_steps, num_sim_steps, L, beta,
                      store_results_every=1,
                      calc_accept_frac_every=None,
                      show_progress=True):
    
    if calc_accept_frac_every is None:
        calc_accept_frac_every = num_sim_steps + num_eq_steps
    
    # Initialize the grid
    grid = ([[0.]*(L+2)] + [[0.] + 
            [2.*random.randint(0, 1)-1 for _ in range(L)] + [0] for _ in range(L)] +
            [[0.]*(L+2)])
    energy = ising_total_energy(grid)
    grid_sum = get_grid_sum(grid)
    
    # Track accept/reject
    accept_fraction_list = []
    accept_fraction_steps = []
    num_accept = 0
    num_reject = 0
    
    # Numpy array to store states
    states_array = np.zeros((num_sim_steps // store_results_every, L, L), dtype=np.float16)
    current_state = 0
    
    # Do the simulations
    step_iter = range(num_eq_steps + num_sim_steps)
    if show_progress:
        step_iter = tqdm(step_iter)
    for step in step_iter:
        
        # Flip each particle
        for i in range(L):
            
            x = i + 1
            for j in range(L):
                
                y = j + 1

                # 10% chance of doing nothing, to break periodicity
                if random.random() <= 0.9:
                
                    # Calculate energy diff
                    energy_diff = (grid[x-1][y] + grid[x][y-1] + 
                           grid[x+1][y] + grid[x][y+1]) * grid[x][y] * 2
                    
                    # Accept or reject the flip
                    if energy_diff < 0 or math.exp(-beta*energy_diff) > random.random():
                        num_accept += 1
                        grid[x][y] *= -1
                        energy += energy_diff
                        grid_sum += 2*grid[x][y]
                    else:
                        num_reject += 1
                
        # Possibly count accept/rejects
        if (step+1) % calc_accept_frac_every == 0:
            accept_fraction_steps.append(step)
            accept_fraction_list.append(num_accept / (num_accept + num_reject))
            num_accept = 0

        # Possibly store the state
        if step >= num_eq_steps and (step - num_eq_steps) % store_results_every == 0:
            states_array[current_state] = np.array(grid)[1:-1, 1:-1]
            current_state += 1

    assert current_state == states_array.shape[0]
            
    # Make return dictionary
    out = dict(accept_fraction_list=accept_fraction_list,
               accept_fraction_steps=accept_fraction_steps,
               states=states_array)
    return out


def binarize_states_array(arr):
    """
    Takes an ising array of (-1, 1) and converts it to (0, 1)
    """
    return (arr > 0).astype(np.float32)

def debinarize_states_array(arr):
    return arr*2.-1.

