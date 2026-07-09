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


class IoTModel:
    """Remote transmission-scheduling MDP (IoT sensor) with centered binomial noise.

    State: sync error s in {-B, ..., B} (clipped random-walk error).
    Action: a in {0, 1} (1 = transmit/reset, 0 = idle).
    Dynamics: S' = clip(S + W) if a = 0, S' = clip(W) if a = 1,
    where W = D - n/2 and D ~ Binomial(n, q).
    Per-step cost: c(s, a) = lam * a + (1 - a) * s**2.
    """

    def __init__(self, B=100, gamma=0.75, n=10, q=0.4, lam=100.0):
        self.B = B
        self.gamma = gamma
        self.n = n
        self.q = q
        self.lam = lam

        self.num_states = 2 * B + 1
        self.num_actions = 2
        self.states = np.linspace(-B, B, self.num_states)
        self.W_support = np.arange(-n // 2, n // 2 + 1)  # noise values W = D - n/2
        self.W = self._binom()

    def _binom(self):
        """Compute binomial PMF of D (noise is W = D - n/2)."""
        P_w = np.zeros(self.n + 1)
        for k in range(len(P_w)):
            P_w[k] = (self.q ** k * (1 - self.q) ** (self.n - k)
                       * math.factorial(self.n)
                       / (math.factorial(k) * math.factorial(self.n - k)))
        return P_w

    def h(self, s):
        """Per-state distortion cost (scalar)."""
        return s ** 2

    def h_vec(self, s_arr):
        """Vectorized distortion cost."""
        return s_arr ** 2

    def cost(self, s_arr, a):
        """Vectorized per-step cost for action a in {0, 1}."""
        return self.lam * a + (1 - a) * self.h_vec(s_arr)
