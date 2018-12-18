"""
This class provides an exact restricted Boltzmann machine implementation in numpy
"""
import numpy as np
import itertools

"""
Functions for enumerating over states, to help calculate the partition function
"""
def numpy_binary_combinations(n):
    v = np.zeros(n)
    yield from _binary_comb_recurse(v, 0)

def _binary_comb_recurse(v, current_pos):
    if current_pos == len(v):
        yield v
    else:
        orig = v[current_pos]
        for e in [0., 1.]:
            v[current_pos] = e
            yield from _binary_comb_recurse(v, current_pos+1)
        v[current_pos] = orig
        
def sigmoid(x):
    """
    This is a useful function for calculating conditional probabilities in an RBM!
    """
    return 1/(1+np.exp(-x))


class RBM:
    """
    Class for a restricted Boltzmann machine. Note that slow methods
    are prefixed by a _ (and shouldn't be called in practical usage).
    
    The goal of this implementation is to explore RBMs from an exact
    standpoint, and learn the math behind their gradients!
    """
    def __init__(self, n_v, n_h, *, b=None, c=None, w=None):
        """
        Initializes an RBM with n_v visible units and n_h hidden units
        
        The values of b,c, and w can be provided
        
        :param n_v: number of visible units
        :param n_h: number of hidden units
        :param b: b vector (visible unit bias)
        :param c: c vector (hidden unit bias)
        :param w: w matrix (for weighting)
        """
        if b is not None:
            assert b.shape == (n_v,)
            self.b = b
        else:
            self.b = np.zeros(n_v)
            
        if c is not None:
            assert c.shape == (n_h,)
            self.c = c
        else:  
            self.c = np.zeros(n_h)
            
        if w is not None:
            assert w.shape == (n_h, n_v)
            self.w = w
        else:
            self.w = np.zeros((n_h, n_v))
        
        self.z = None # avoid calculating partition function
        
    def energy(self, v, h):
        """
        Returns the "energy" of an RBM, defined as
        - h^T W v - vb - hc (all products are dot products)
        """
        term_1 = np.dot(h, self.w @ v)
        term_2 = np.dot(h, self.c)
        term_3 = np.dot(v, self.b)
        return -term_1 - term_2 - term_3
    
    def get_prob(self, v, h, normalized=False):
        """
        Returns the probability of a given state,
        which is proportional to exp(- energy) of the state.
        
        It can be optionally normalized with the partition function
        """
        p = np.exp(-self.energy(v, h))
        if normalized:
            p /= self.z
        return p
    
    def get_cond_prob(self, *, v=None, h=None):
        """
        Compute p(x|y), where y is the kwarg specified, and x is the one not
        """
        assert not( v is None and h is None)
        if v is not None:
            return sigmoid(self.w @ v + self.c)
        if h is not None:
            return sigmoid(self.w.T @ h + self.b)
        else:
            raise RuntimeError("Either v or h must be specified")
            
    def gibbs_sample_v(self, v_start, k):
        """
        Does Gibbs sampling from this RBM to get a likely state v
        after k sampling steps.
        """
        v_current = v_start
        h_current = None
        for i in range(k):
            h_prob = self.get_cond_prob(v=v_current)
            h_current = (np.random.random(size=self.c.shape) < h_prob).astype(v_start.dtype)
            
            v_prob = self.get_cond_prob(h=h_current)
            v_current = (np.random.random(size=self.b.shape) < v_prob).astype(v_start.dtype)
            
        return v_current

    def gibbs_sample_array(self, n_samples, k):
        """
        Draws a large number of samples from this RBM's distribution
        using a large, repeated Gibbs chain
        """
        samples = np.zeros((n_samples, len(self.b)))
        v_start = np.zeros_like(self.b)
        for current_sample in range(n_samples):
            v_out = self.gibbs_sample_v(v_start, k=k)
            v_start = v_out
            samples[current_sample] = v_start
        return samples
        
    def apply_gradients(self, grad_dict, *, learning_rate, weight_decay = 0):
    
        # Apply weight decay
        if weight_decay > 0:
            self.w -= self.w * weight_decay
            self.b -= self.b * weight_decay
            self.c -= self.c * weight_decay
            
        # Apply the gradients
        self.w += learning_rate * grad_dict["w"]
        self.b += learning_rate * grad_dict["b"]
        self.c += learning_rate * grad_dict["c"]      
    
    def get_contrastive_divergence(self, v_given, k):
        """
        Returns the contrastive divergence "gradients",
        which estimate the true gradients via Gibbs sampling
        """
        grads = dict(w=np.zeros_like(self.w), c=np.zeros_like(self.c), b=np.zeros_like(self.b))
        
        # Calculate the gradient for the data
        grads["w"] += np.outer(self.get_cond_prob(v=v_given), v_given)
        grads["b"] += v_given
        grads["c"] += self.get_cond_prob(v=v_given)
        
        # Get gradient for the model
        v_gibbs = self.gibbs_sample_v(v_given, k)
        grads["w"] -= np.outer(self.get_cond_prob(v=v_gibbs), v_gibbs)
        grads["b"] -= v_gibbs
        grads["c"] -= self.get_cond_prob(v=v_gibbs)
        
        return grads
            
    def __str__(self):
        return "RBM: w=\n{}\nb=\n{}\nc=\n{}\n".format(self.w, self.b, self.c)
    
    def __repr__(self):
        return str(self)
            
    def _get_z(self):
        """
        Calculates the partition function of this RBM.
        
        It runs in exponential time, so call this function sparingly!!
        """
        running_z = 0.
        for v in numpy_binary_combinations(len(self.b)):
            for h in numpy_binary_combinations(len(self.c)):
                running_z += self.get_prob(v, h, normalized=False)
        
        self.z = running_z
        return running_z
            
    def _get_marginalized_prob(self, *, v=None, h=None, normalized=True):
        """
        Compute p(x), where x is the kwarg not specified
        
        Done by summing up over all the other states (so very inefficient)
        """
        if v is not None:
            return sum([self.get_prob(v, h_sum, normalized=normalized) 
                        for h_sum in numpy_binary_combinations(len(self.c))])
        if h is not None:
            return sum([self.get_prob(v_sum, h, normalized=normalized) 
                        for v_sum in numpy_binary_combinations(len(self.b))])
        else:
            raise RuntimeError("Must provide either v or h.")
            
    def _get_exact_gradients(self, v_dist_dict):
        """
        Computes the gradient of the expected log likelihood of a given
        distribution v_dist_dict.
        
        Returns them as a dictionary of gradients
        """
        
        # Calculate the partition function
        self._get_z()
        grads = dict(w=np.zeros_like(self.w), c=np.zeros_like(self.c), b=np.zeros_like(self.b))
        for v in numpy_binary_combinations(len(self.b)):
            p_h_given_v = self.get_cond_prob(v=v)
            qv = v_dist_dict[tuple(v)]
            
            # Gradient of b (no dependence on h)
            grads["b"] += qv * v  - v * self._get_marginalized_prob(v=v)
            
            for h in numpy_binary_combinations(len(self.c)):
                
                # Takes prob if 1, 1-prob if 0
                p_this_h_given_v = h * p_h_given_v + (1-h)*(1-p_h_given_v)
                
                # w gradient (times 2 probabilities)
                w_grad = np.outer(h, v) * (np.prod(p_this_h_given_v) * qv - 
                                           self.get_prob(v, h, normalized=True))
                grads["w"] += w_grad
                
                # term c gradient
                grads["c"] += h * (np.prod(p_this_h_given_v) * qv - self.get_prob(v, h, normalized=True))
                
        return grads
    
    def _get_exact_single_sample_gradients(self, v_given):
        """
        Computes the gradient of the expected log likelihood of a given
        sample of v, v_given.
        
        Still computationally inefficient, as it computes the expectation
        over the RBM's distribution (taking exponential time)
        
        Returns them as a dictionary of gradients
        """
        grads = dict(w=np.zeros_like(self.w), c=np.zeros_like(self.c), b=np.zeros_like(self.b))
        
        # Calculate the gradient for the data
        grads["w"] += np.outer(self.get_cond_prob(v=v_given), v_given)
        grads["b"] += v_given
        grads["c"] += self.get_cond_prob(v=v_given)
        
        # Now the model gradient (the hard part)
        self._get_z()
        for v in numpy_binary_combinations(len(self.b)):
            p_h_given_v = self.get_cond_prob(v=v)
            
            grads["b"] -= v * self._get_marginalized_prob(v=v)
            for h in numpy_binary_combinations(len(self.c)):
                p_this_h_given_v = h * p_h_given_v + (1-h)*(1-p_h_given_v)
                grads["w"] -= np.outer(h, v) * self.get_prob(v, h, normalized=True)
                grads["c"] -= h * self.get_prob(v, h, normalized=True)
                
        return grads
        
if __name__ == "__main__":

    from tqdm import trange
    print("Running some RBM Tests")
    np.random.seed(203)
    rbm = RBM(2, 3, w=np.random.randn(3, 2))
    print(rbm)
    
    # Try to learn the distribution to always output (0., 1.)
    # Compute the (computationally expensive) conditional probability of (0., 1.)
    v_desired = np.array([0., 1.])
    rbm._get_z()
    pv_start = rbm._get_marginalized_prob(v=v_desired)
    print("Start prob: {}".format(pv_start))
    learning_rate = 1e-2
    print("Training...")
    for i in trange(10000):
        grads = rbm.get_contrastive_divergence(v_given=v_desired, k=3)

        rbm.w += learning_rate * grads["w"]
        rbm.b += learning_rate * grads["b"]
        rbm.c += learning_rate * grads["c"]

    # Now get marginalized probability after training
    rbm._get_z()
    pv_end = rbm._get_marginalized_prob(v=v_desired)
    print("End prob: {}".format(pv_end))
    
    
    
    # Try to learn 1. with 26% probability, 0. with 74%
    print("Learn a 1x1 RBM")
    rbm = RBM(1, 1, w=np.random.randn(1,1))
    v_dist_target = {(1.0,): 0.26, (0.0,): 0.74}
    print(rbm)
    rbm._get_z()
    for v in v_dist_target.keys():
        print("{}: {}".format(v, rbm._get_marginalized_prob(v=np.array(v))))
    learning_rate = 1e-2
    print("Training...")
    for i in trange(10000):
        grads = rbm._get_exact_gradients(v_dist_target)
        rbm.apply_gradients(grads, learning_rate=learning_rate)

    # Now get marginalized probability after training
    rbm._get_z()
    print("Training done")
    for v in v_dist_target.keys():
        print("{}: {}".format(v, rbm._get_marginalized_prob(v=np.array(v))))
    print(rbm)
    print(rbm.z)

