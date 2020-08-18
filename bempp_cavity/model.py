"""

"""
from abc import abstractmethod
import time

import numpy as np

import bempp.api
from bempp.api import assembly, linalg
from bempp.api.operators.boundary import maxwell, sparse

from .plotting import strong_form_plot


SOLVER_OPTIONS = dict(
    restart=500,
    maxiter=3000
)


class Model:
    """
    todo
    """
    def __init__(self,
            cavity_grid, wave_numbers, mu_numbers, wave,
            spaces='default'
        ):
        self.system = None
        self.preconditioner = None

        if spaces == 'default':
            self.system = DefaultSystem(
                    cavity_grid, wave_numbers, mu_numbers, wave)
        elif spaces == 'RWG-dominant':
            self.system = RWGDominantSystem(
                    cavity_grid, wave_numbers, mu_numbers, wave)


    def solve(self, **kwargs):
        """
        todo
        """
        return self.system.solve(**kwargs)


class System:
    def __init__(self, cavity_grid, wave_numbers, mu_numbers, wave):
        self.cavity_grid = cavity_grid
        self.wave_numbers = wave_numbers
        self.mu_numbers = mu_numbers
        self.wave = wave

        self.operator = self.assemble_operator()
        self.rhs      = self.assemble_rhs()
    
    @abstractmethod
    def get_ops(self, *args):
        pass
    
    def assemble_operator(self):
        """
        todo
        """
        Aw_1, A1_1, Aw_1w, Aw_w1, Aw_w, Ae_w = self.get_ops()

        A = assembly.BlockedOperator(2 * 2, 2 * 2)
        assign_in_place_subblock(A, -(Aw_1 + A1_1), 0, 0)
        assign_in_place_subblock(A,   Aw_1w,        0, 1)
        assign_in_place_subblock(A, - Aw_w1,        1, 0)
        assign_in_place_subblock(A,   Aw_w + Ae_w,  1, 1)
        return A
    
    @abstractmethod
    def assemble_rhs(self, *args):
        pass
    
    @abstractmethod
    def solve(self):
        pass 

    def get_memory_size(self):
        """
        todo
        """
        ops = self.get_ops()
        memory = 0
        for op in ops:
            for i in range(2):
                for j in range(2):
                    memory += op[i, j].strong_form().memory
        return int(memory)


class DefaultSystem(System):
    """
    todo
    """
    def __init__(self, *args):
        super(DefaultSystem, self).__init__(*args)
    
    def get_ops(self):
        """
        todo
        """
        k = self.wave_numbers
        small_grid = self.cavity_grid.cavities[0]
        large_grid = self.cavity_grid.main
        # small cube
        A1_1 = maxwell.multitrace_operator(small_grid, k[2])
        Aw_1 = maxwell.multitrace_operator(small_grid, k[1])
        # large cube
        Aw_w = maxwell.multitrace_operator(large_grid, k[1])
        Ae_w = maxwell.multitrace_operator(large_grid, k[0])
        # mixed 
        Aw_1w = maxwell.multitrace_operator(large_grid, k[1], target=small_grid) 
        Aw_w1 = maxwell.multitrace_operator(small_grid, k[1], target=large_grid) 

        return Aw_1, A1_1, Aw_1w, Aw_w1, Aw_w, Ae_w

    def assemble_rhs(self):
        """
        todo
        """
        I = sparse.multitrace_identity(self.cavity_grid.main, spaces='maxwell')

        Aw_w = maxwell.multitrace_operator(self.cavity_grid.main, self.wave_numbers[1])

        Aw_1w = maxwell.multitrace_operator(
            self.cavity_grid.main, self.wave_numbers[1],
            target=self.cavity_grid.cavities[0])

        dTrace = self.wave.dirichlet_trace(self.operator.domain_spaces[2])
        nTrace = self.wave.neumann_trace(self.operator.domain_spaces[3])

        rhs = [
            - Aw_1w * [dTrace, nTrace],
            - (Aw_w - 1/2 * I) * [dTrace, nTrace]
        ]

        rhs = [rhs[0][0], rhs[0][1], rhs[1][0], rhs[1][1]] # flatten
        return rhs
    
    def solve(self, preconditioner='none', tol=1e-5):
        """
        todo
        """
        if preconditioner == 'none':
            super_operator = self.operator
            super_rhs = self.rhs
        elif preconditioner == 'diagonal':
            diagonal = self.get_diagonal() 
            super_operator = diagonal * self.operator
            super_rhs = diagonal * self.rhs
        elif preconditioner == 'self':
            super_operator = self.operator * self.operator
            super_rhs = self.operator * self.rhs
        else:
            raise NotImplementedError(
                "Preconditioner '%s' not supported" % preconditioner)

        time_assemble = -time.clock() # start timer
        super_operator.strong_form(True)
        time_assemble += time.clock() # stop timer

        bempp.api.MATVEC_COUNT = 0 # reset the MATVEC counter to 0
        solve_time = -time.clock() # initialise the timer
        sol, solve_info, residuals = linalg.gmres(
            super_operator, super_rhs,
            tol=tol,
            use_strong_form=True,
            return_residuals=True,
            **SOLVER_OPTIONS
        )
        solve_time += time.clock() # stop the timer
        matvec_count = bempp.api.MATVEC_COUNT # sample the matvec counter
        
        info = dict(
            status=solve_info,
            time_solve=solve_time,
            time_assemble=time_assemble,
            matvec_count=matvec_count
        )
        return Solution(traces=sol, info=info, residuals=residuals, system=self)
    
    def get_diagonal(self):
        """
        todo
        """
        Aw_1, A1_1, _, _, Aw_w, Ae_w = self.get_ops()
        D = assembly.BlockedOperator(2 * 2, 2 * 2)
        assign_in_place_subblock(D, -(Aw_1 + A1_1), 0, 0)
        assign_in_place_subblock(D,   Aw_w + Ae_w,  1, 1)
        return D

        
class RWGDominantSystem(System):
    """
    todo
    """
    def __init__(self, *args):
        super(RWGDominantSystem, self).__init__(*args)

    def get_ops(self):
        """
        todo
        """
        # return Aw_1, A1_1, Aw_1w, Aw_w1, Aw_w, Ae_w
        pass


def assign_in_place_subblock(A, a, i, j):
    """
    Assigns the 4 elements of a to the 2x2 block of A
    specified by the indexes i and j.
    """
    bi = 2*i
    bj = 2*j
    A[bi, bj]     = a[0, 0]
    A[bi, bj+1]   = a[0, 1]
    A[bi+1, bj]   = a[1, 0]
    A[bi+1, bj+1] = a[1, 1]


class Solution:
    """
    todo
    """
    def __init__(self,
            coefficients=None, traces=None, info=None, system=None,
            residuals=None
        ):
        assert coefficients is not None or traces is not None, \
            """Either coefficients or traces must be supplied to the
            Solutions constructor"""

        self._coefficients = coefficients
        self._traces = traces
        self.info = info
        self.system = system
        self.residuals = residuals

    @property
    def coefficients(self):
        if self._coefficients is None:
            return np.hstack([c.coefficients for c in self._traces])
        return self._coefficients
    
    @property
    def traces(self):
        if self._traces is None:
            raise NotImplementedError("Weak implementation not supported")
        else:
            return self._traces

    def plot(self):
        """
        todo
        """
        if self._traces is not None:
            strong_form_plot(self)



