"""

"""
from abc import abstractmethod
import time

import numpy as np

import bempp.api
from bempp.api import assembly, linalg, function_space
from bempp.api.operators.boundary import maxwell, sparse
from .login import gmres as login_gmres

from .plotting import strong_form_plot, weak_form_plot


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
        else:
            raise NotImplementedError("'%s' is not supported" % spaces)


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
        self.use_strong_form = None
   
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
    
    def solve(self, preconditioner='none', tol=1e-5):
        """
        todo
        """
        time_assemble = -time.clock() # start timer

        self.operator = self.assemble_operator()
        self.rhs      = self.assemble_rhs()

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
        # elif preconditioner == 'electric-interior':
        #     pass
        else:
            raise NotImplementedError(
                "Preconditioner '%s' not supported" % preconditioner)

        if hasattr(super_operator, 'strong_form'):
            super_operator.strong_form(True)
            
        time_assemble += time.clock() # stop timer
        

        bempp.api.MATVEC_COUNT = 0 # reset the MATVEC counter to 0
        solve_time = -time.clock() # initialise the timer
        sol, solve_info, residuals = self._gmres(super_operator, super_rhs, tol)
        solve_time += time.clock() # stop the timer
        matvec_count = bempp.api.MATVEC_COUNT # sample the matvec counter
        
        info = dict(
            status=solve_info,
            time_solve=solve_time,
            time_assemble=time_assemble,
            matvec_count=matvec_count
        )
        if isinstance(sol[0], np.complex128):
            return Solution(coefficients=sol, info=info, residuals=residuals, system=self)            
        else:
            return Solution(traces=sol, info=info, residuals=residuals, system=self)
    
    @abstractmethod
    def get_diagonal(self):
        pass

    @abstractmethod
    def get_as_operator(self, op):
        pass

    @abstractmethod
    def _gmres(self, *args):
        pass


class DefaultSystem(System):
    """
    todo
    """
    def __init__(self, *args):
        super(DefaultSystem, self).__init__(*args)
        self.use_strong_form = True
    
    def _gmres(self, super_operator, super_rhs, tol):
        """
        todo
        """
        sol, solve_info, residuals = linalg.gmres(
            super_operator, super_rhs,
            tol=tol,
            use_strong_form=True,
            return_residuals=True,
            **SOLVER_OPTIONS
        )
        return sol, solve_info, residuals

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

    def get_diagonal(self):
        """
        todo
        """
        Aw_1, A1_1, _, _, Aw_w, Ae_w = self.get_ops()
        D = assembly.BlockedOperator(2 * 2, 2 * 2)
        assign_in_place_subblock(D, -(Aw_1 + A1_1), 0, 0)
        assign_in_place_subblock(D,   Aw_w + Ae_w,  1, 1)
        return D
    
    def get_as_operator(self, op):
        """
        todo
        """
        return op.strong_form

        
class RWGDominantSystem(System):
    """
    todo
    """
    def __init__(self, *args):
        methods = [
            'RWG',
            # 'B-RWG'
            'BC',
            'SNC'
            # 'B-SNC'
            # 'RBC'
        ]
        self.main = discretize(args[0].main, *methods)
        self.cavities = [discretize(grid, *methods) for grid in args[0].cavities]
        super(RWGDominantSystem, self).__init__(*args)
        self.use_strong_form = False

    def get_ops(self):
        """
        todo
        """
        k = self.wave_numbers
        mu = self.mu_numbers[0]
        domain = 'RWG'
        range_ = 'RWG'
        dtr = 'SNC'
        # small cube
        A1_1 = get_simple_block_op(self.cavities[0], k[2], mu, domain, range_, dtr)
        Aw_1 = get_simple_block_op(self.cavities[0], k[1], mu, domain, range_, dtr)
        # large cube
        Aw_w = get_simple_block_op(self.main, k[1], mu, domain, range_, dtr)
        Ae_w = get_simple_block_op(self.main, k[0], mu, domain, range_, dtr)
        # mixed (target, source, k)
        Aw_1w = get_mixed_block_op(self.cavities[0], self.main, k[1], mu, domain, range_, dtr)
        Aw_w1 = get_mixed_block_op(self.main, self.cavities[0], k[1], mu, domain, range_, dtr)
   
        return Aw_1, A1_1, Aw_1w, Aw_w1, Aw_w, Ae_w
    
    def get_as_operator(self, op):
        """
        todo
        """
        return op
    
    def assemble_operator(self):
        """
        todo
        """
        operator = super(RWGDominantSystem, self).assemble_operator()
        return operator.weak_form()
    
    def assemble_rhs(self):
        """
        todo
        """
        DOMAIN_OP = 'RWG'
        RANGE_OP = 'RWG'
        DTR_OP = 'SNC'
        large_cube = self.main
        small_cube = self.cavities[0]
        KW = self.wave_numbers[1]
        MU = self.mu_numbers[0]

        Aw_1w = get_mixed_block_op(
            small_cube, large_cube,
            KW, MU,
            DOMAIN_OP, RANGE_OP, DTR_OP
        )

        Aw_w  = get_simple_block_op(
            large_cube, KW, MU,
            DOMAIN_OP, RANGE_OP, DTR_OP                       
        )

        I = self.manually_get_block_identity_op(
            large_cube, DOMAIN_OP, RANGE_OP, DTR_OP)

        pre = [
            - Aw_1w.weak_form() * self.wave.coefficients(large_cube[DOMAIN_OP]),
            - (Aw_w.weak_form() - 1/2 * I.weak_form()) * self.wave.coefficients(large_cube[DOMAIN_OP])
        ]
        # flatten list of lists
        b1 = [y for x in pre for y in x]
        return b1

    def get_identity_op(self, space, domain, range_, dtr):
        """
        Shortcut to the sparse identity operator.
        """
        return bempp.api.operators.boundary.sparse.identity(
            # this can make the kernel crash if not set correctly
            space[domain], space[range_], space[dtr]
        )
    
    def manually_get_block_identity_op(self, space, domain, range_, dtr):
        """
        Create an identity operator matching the given space.
        """
        i = self.get_identity_op(space, domain, range_, dtr)
        I = assembly.BlockedOperator(2, 2)
        I[0, 0] = i
        I[1, 1] = i
        return I

    def _gmres(self, super_operator, super_rhs, tol):
        """
        todo
        """
        return login_gmres(
            super_operator, super_rhs, tol,
            return_residuals=True,
            **SOLVER_OPTIONS
        )
    
    def get_diagonal(self):
        """
        todo
        """
        # D = assembly.BlockedOperator(2 * 2, 2 * 2)
        # for i in range(4):
        #     D[i, i] = self.operator[i, i]
        # return D.weak_form()
        Aw_1, A1_1, _, _, Aw_w, Ae_w = self.get_ops()
        D = assembly.BlockedOperator(2 * 2, 2 * 2)
        assign_in_place_subblock(D, -(Aw_1 + A1_1), 0, 0)
        assign_in_place_subblock(D,   Aw_w + Ae_w,  1, 1)
        return D.weak_form()
        

def discretize(grid, *methods_list):
    """
    Returns a named tuple containing the discretised boundaries of the
    given grid according to the method specified.
    """
    space = {}
    methods = list(set(methods_list))
    assert len(methods) > 0, "You must provide disretisation methods."
    for method_key in methods:
        method = method_key.split(".")[0]
        space[method_key] = function_space(grid, method, 0)
    return space


def to_block_op(mfie, efie, k, mu):
    """
    Build the standard block operator from the given integral equations.
    """
    A = assembly.BlockedOperator(2, 2) # empty operator object
    A[0,0] = mfie
    A[0,1] = mu/k * efie
    A[1,0] = -k/mu * efie
    A[1,1] = mfie
    return A


def get_simple_block_op(space, k, mu, domain, range_, dtr):
    """
    Return a 2x2 block operator defining the block matrix that would
    act on the given grid.
    
    This is similar to the `multitrace_operator` constructor, but it
    allows us to specify exactly which boundary disretisation functions
    to use.
    """
    efie = maxwell.electric_field(
        space[domain], space[range_], space[dtr], k,
    )
    mfie = maxwell.magnetic_field(
        space[domain], space[range_], space[dtr], k,
    )
    A = to_block_op(mfie, efie, k, mu)
    return A


def get_mixed_block_op(target, source, k, mu, domain, range_, dtr):
    """
    Return a 2x2 block operator that defines the interferences on
    `grid_a` by `grid_b`.
    """
    efie = maxwell.electric_field(
        source[domain], target[range_], target[dtr], k
    )
    mfie = maxwell.magnetic_field(
        source[domain], target[range_], target[dtr], k
    )    
    A = to_block_op(mfie, efie, k, mu)
    return A


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
        else:
            weak_form_plot(self)

    def get_total_memory_size(self):
        """
        todo
        """
        memory = 0
        for i in range(4):
            for j in range(4):
                memory += self.system.operator[i, j].memory
        return memory
