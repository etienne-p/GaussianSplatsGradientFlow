import numpy as np


class AdamOptimizer:
    """Stateful Adam optimizer. Call step() once per parameter per gradient step."""

    # Using usual defaults from https://arxiv.org/abs/1412.6980
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._state = {}

    def step(self, name, param, grad):
        # One state entry per parameter, instantiated lazily.
        if name not in self._state:
            self._state[name] = (np.zeros_like(param), np.zeros_like(param), 0)
        m, v, t = self._state[name]
        # Adam update rule: https://arxiv.org/abs/1412.6980
        t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        self._state[name] = (m, v, t)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self._state.clear()
