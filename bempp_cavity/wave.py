"""
This file contains the Wave class.
"""

import numpy as np
from bempp.api import GridFunction

class Wave:
    """
    Parent class on which to base future wave types.
    """
    pass


class IncidentWave(Wave):
    """
    The Incident Wave is passed to the Model object.
    It can get the value of its trace on specified boundaries.
    """
    def __init__(self, k, mu):
        self.k = k
        self.mu = mu
    
    def neumann_trace(self, space):
        """
        Return the neumann trace GridFunction on the specified space
        """
        return self.k/self.mu * GridFunction(
            space,
            fun=self._neumann_trace_fun
        )

    def dirichlet_trace(self, space):
        """
        Return the dirichlet trace GridFunction on the specified space
        """
        return GridFunction(
            space,
            fun=self._dirichlet_trace_fun
        )

    def coefficients(self, space):
        """
        Return the trace data as a list of coefficients.
        """
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
