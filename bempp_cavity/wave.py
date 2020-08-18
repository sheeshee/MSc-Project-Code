"""
todo
"""

import numpy as np
from bempp.api import GridFunction

class Wave:
    pass


class IncidentWave(Wave):
    """
    todo
    """
    def __init__(self, k, mu):
        self.k = k
        self.mu = mu
        print(k, mu)
    
    def neumann_trace(self, space):
        return self.k/self.mu * GridFunction(
            space,
            fun=self._neumann_trace_fun
        )

    def dirichlet_trace(self, space):
        return GridFunction(
            space,
            fun=self._dirichlet_trace_fun
        )

    def coefficients(self, space):
        return self.dirichlet_trace(space).coefficients.tolist() + \
            self.neumann_trace(space).coefficients.tolist()

    def incident_field(self, x):
        return np.array([0. * x[2], 0. * x[2], np.exp(1j * self.k * x[0])])

    def _dirichlet_trace_fun(self, x, n, domain_index, result):
        result[:] = np.cross(self.incident_field(x), n)
 
    def _curl(self, incident_field, x):
        return np.array([0,  - 1j * self.k * np.exp(1j * self.k * x[0]), 0])

    def _neumann_trace_fun(self, x, n, domain_index, result):
        result[:] = (1/(1j * self.k)) * np.cross(self._curl(self.incident_field,x), n)
