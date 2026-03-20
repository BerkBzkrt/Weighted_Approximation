import math
import numpy as np


class InventoryMDP:
    """Inventory management MDP with binomial demand and linear holding/shortage costs."""

    def __init__(self, s_max=500, gamma=0.75, n=10, q=0.4, ch=4, cs=2, p=5):
        self.s_max = s_max
        self.gamma = gamma
        self.n = n
        self.q = q
        self.ch = ch
        self.cs = cs
        self.p = p

        self.num_states = 2 * s_max + 1
        self.num_actions = s_max + 1
        self.states = np.linspace(-s_max, s_max, self.num_states)
        self.W = self._binom()

    def _binom(self):
        """Compute binomial demand PMF."""
        P_w = np.zeros(self.n + 1)
        for k in range(len(P_w)):
            P_w[k] = (self.q ** k * (1 - self.q) ** (self.n - k)
                       * math.factorial(self.n)
                       / (math.factorial(k) * math.factorial(self.n - k)))
        return P_w

    def h(self, s):
        """Per-state holding/shortage cost (scalar)."""
        if s >= 0:
            return self.ch * s
        else:
            return -self.cs * s

    def h_vec(self, s_arr):
        """Vectorized holding/shortage cost."""
        return np.where(s_arr >= 0, self.ch * s_arr, -self.cs * s_arr)
