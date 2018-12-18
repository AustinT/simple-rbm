"""
This class provides an exact restricted Boltzmann machine implementation in pytorch.

It is intended to be used for gradient comparison
"""
import torch
from rbm import *

"""
Convenience function for converting to/from tensors
"""
def torch_convert(v):
    return torch.tensor(v, dtype=torch.float)

class TorchRBM(RBM):
    
    def __init__(self, n_v, n_h, *, b=None, c=None, w=None):
        """
        Initializes an RBM with n_v visible units and n_h hidden units
        
        The values of b,c, and w can be provided (as numpy arrays)
        
        :param n_v: number of visible units
        :param n_h: number of hidden units
        :param b: b vector (visible unit bias)
        :param c: c vector (hidden unit bias)
        :param w: w matrix (for weighting)
        """
        if b is not None:
            assert b.shape == (n_v,)
            self.b = torch_convert(b)
            self.b.requires_grad_(True)
        else:
            self.b = torch.zeros(n_v, requires_grad=True)
            
        if c is not None:
            assert c.shape == (n_h,)
            self.c = torch_convert(c)
            self.c.requires_grad_(True)
        else:  
            self.c = torch.zeros(n_h, requires_grad=True)
            
        if w is not None:
            assert w.shape == (n_h, n_v)
            self.w = torch_convert(w)
            self.w.requires_grad_(True)
        else:
            self.w = torch.zeros((n_h, n_v), requires_grad=True)
        
        self.z = None # avoid calculating partition function
        
    def energy(self, v, h):
        """
        Returns the "energy" of an RBM, defined as
        - h^T W v - vb - hc (all products are dot products)
        
        v and h can be numpy arrays
        """
        v = torch_convert(v)
        h = torch_convert(h)
        term_1 = torch.dot(torch.mv(self.w, v), h)
        term_2 = torch.dot(h, self.c)
        term_3 = torch.dot(v, self.b)
        return -term_1 - term_2 - term_3
    
    def get_prob(self, v, h, normalized=False):
        p = torch.exp(-self.energy(v, h))
        if normalized:
            p = p / self.z
        return p
    
    def get_cond_prob(self, *, v=None, h=None):
        """
        Compute p(x|y), where y is the kwarg specified, and x is the one not
        returns probability that x is 1
        """
        assert not( v is None and h is None)
        if v is not None:
            v = torch_convert(v)
            return torch.sigmoid(torch.mv(self.w, v) + self.c)
        if h is not None:
            h = torch_convert(h)
            return torch.sigmoid(torch.mv(self.w.t(), h) + self.b)
        else:
            raise RuntimeError
            
    def _get_marginalized_prob(self, *, v=None, h=None, normalized=True):
        assert (v is None) != (h is None)
        if v is not None:
            v = torch_convert(v)
            return sum([self.get_prob(v, h_sum, normalized=normalized) 
                        for h_sum in numpy_binary_combinations(len(self.c))])
        if h is not None:
            return sum([self.get_prob(v_sum, h, normalized=normalized) 
                        for v_sum in numpy_binary_combinations(len(self.b))])
        else:
            raise RuntimeError("Must provide either v or h.")
    
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
    
    def _get_log_likelihood(self, *, v=None, v_dist_dict=None):
        self._get_z()
        
        # Calculate for a single vector or a distribution, but not both
        assert (v is None) != (v_dist_dict is None)
        if v is not None:
            return torch.log(self._get_marginalized_prob(v=v, normalized=True))
        
        # In this case, calculate for the distribution
        ll = 0.
        for v in numpy_binary_combinations(len(self.b)):
            qv = v_dist_dict[tuple(v)]
            ll = ll + qv * torch.log(self._get_marginalized_prob(v=v, normalized=True))

        return ll
    
    def _get_exact_gradients(self, v_dist_dict):
        raise NotImplementedError        
        
    def gibbs_sample_v(self, v_start, k):
        raise NotImplementedError
    
    def get_contrastive_divergence(self, v_given, k):
        raise NotImplementedError
            
    def _get_exact_gradients(self, v_dist_dict):
        raise NotImplementedError
    
    def _get_exact_single_sample_gradients(self, v_given):
        raise NotImplementedError
        
if __name__ == "__main__":
    print("Testing TorchRBM (in comparison to regular RBM")
    
    rbm = RBM(2, 3, w=np.random.randn(3, 2))
    torch_rbm = TorchRBM(2, 3, w=rbm.w)
    print("RBM:")
    print(rbm)
    print("Torch RBM:")
    print(torch_rbm)

    print("Testing gradients (should match up)")
    v_test = np.array([1., 0.])
    rbm_grads = rbm._get_exact_single_sample_gradients(v_test)
    log_likelihood = torch_rbm._get_log_likelihood(v=v_test)
    log_likelihood.backward()

    for name, grad in sorted(rbm_grads.items()):
        print(name)
        print("From RBM:")
        print(grad)
        print("From TorchRBM")
        torch_rbm_grad = getattr(torch_rbm, name).grad.numpy().astype(grad.dtype)
        print(torch_rbm_grad)
        print("Total Abs Error (should be low if they match): {:.2e}".format(np.sum(np.abs(torch_rbm_grad - grad))))
        print()


