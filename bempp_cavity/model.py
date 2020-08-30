"""

"""
from abc import abstractmethod
import time

import numpy as np

import bempp.api
from bempp.api import assembly, linalg, function_space
from bempp.api.operators.boundary import maxwell, sparse
from .login import gmres as login_gmres

from .plotting import strong_form_plot, weak_form_plot, show_domains


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
        self._operators = None
        self.main = None
        self.cavities = None
   
    def get_ops(self, *args):
        if self._operators is None:
            self._operators = self._compile_ops()
        return self._operators

    @property
    def N(self):
        return len(self.cavity_grid.cavities) + 1
    
    def assemble_operator(self):
        """
        todo
        """
        N = len(self.cavity_grid.cavities) # system size
        operators = self.get_ops()

        A = assembly.BlockedOperator((N+1) * 2, (N+1) * 2)
        for (row, col), ops_dict in operators.items():
            ops = list(ops_dict.values())
            op_sum = sum(ops[1:], ops[0])
            assign_in_place_subblock(A, op_sum, row, col)

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

    @abstractmethod
    def multitrace_operator(self, *args, **kwargs):
        pass

    def _compile_ops(self):
        """
        todo
        """
        ke = self.wave_numbers[0]
        kw = self.wave_numbers[1]
        ki = self.wave_numbers[2:]
        mu = 1
        cavities = self.cavities
        ops = {}
        def add(i, j, op, key='default'):
            if (i, j) not in ops:
                ops[(i, j)] = {key: op}
            else:
                if key in ops[(i, j)]:
                    raise ValueError("Duplicate key value provided in operator construction")
                else:
                    ops[(i, j)][key] = op

        # cavities
        for row, _ in enumerate(cavities):
            for col, _ in enumerate(cavities):
                if row == col:
                    add(
                        row, col,
                        -1 * self.multitrace_operator(ki[row], mu, cavities[row])
                    )
                    add(
                        row, col,
                        -1 * self.multitrace_operator(kw, mu, cavities[row]),
                        key='wall'
                    )
                else:
                    add(
                        row, col,
                        -1 * self.multitrace_operator(kw, mu, cavities[col], target=cavities[row])
                    ),
            # # self to wall
            add(
                row, col+1,
                self.multitrace_operator(kw, mu, self.main, target=cavities[row])
            )
        
        for col, cavity in enumerate(cavities):
            add(
                row+1, col,
                -1 * self.multitrace_operator(kw, mu, cavity, target=self.main)
            )
        
        # external boundary
        add(
            row+1, col+1,
            self.multitrace_operator(kw, mu, self.main),
            key='wall'

        )
        add(
            row+1, col+1,
            self.multitrace_operator(ke, mu, self.main),
            key='exterior'
        )

        return ops

class DefaultSystem(System):
    """
    todo
    """
    def __init__(self, *args):
        super(DefaultSystem, self).__init__(*args)
        self.use_strong_form = True
        self.main = self.cavity_grid.main
        self.cavities = self.cavity_grid.cavities
    
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

    def multitrace_operator(
            self,
            k, mu, base, target=None
        ):
        """
        todo
        """
        return maxwell.multitrace_operator(base, k, target=target)
        

    def assemble_rhs(self):
        """
        todo
        """
        ops = self.get_ops()
        rhs = [None] * self.N
        dTrace = self.wave.dirichlet_trace(self.operator.domain_spaces[-2])
        nTrace = self.wave.neumann_trace(self.operator.domain_spaces[-1])
        I = sparse.multitrace_identity(self.cavity_grid.main, spaces='maxwell')

        for i in range(self.N - 1):
            rhs[i] = -1 * ops[(i, self.N - 1)]['default'] * [dTrace, nTrace]
        
        rhs[self.N - 1] = -1 * (ops[(self.N - 1, self.N - 1)]['wall'] - 0.5 * I) * [dTrace, nTrace]

        return [a for b in rhs for a in b] # flatten list and return

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
        super(RWGDominantSystem, self).__init__(*args)
        self.main = discretize(args[0].main, *methods)
        self.cavities = [discretize(grid, *methods) for grid in args[0].cavities]
        self.use_strong_form = False
    
    def multitrace_operator(self, k, mu, base, target=None):
        if target is None:
            target = base
        return get_simple_block_op(base, target, k, mu)
    
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
        rhs = [None] * self.N
        ops = self.get_ops()
        # large_cube = self.main
        # small_cube = self.cavities[0]
        # KW = self.wave_numbers[1]
        # MU = self.mu_numbers[0]
        I = self.manually_get_block_identity_op(self.main)
        u_inc = self.wave.coefficients(self.main['RWG'])

        for i in range(self.N - 1):
            rhs[i] = -1 * ops[(i, self.N - 1)]['default'].weak_form() * u_inc
        
        rhs[self.N - 1] = -1 * (ops[(self.N - 1, self.N - 1)]['wall'].weak_form() - 0.5 * I.weak_form()) * u_inc


        # pre = [
        #     - Aw_1w.weak_form() * self.wave.coefficients(large_cube[DOMAIN_OP]),
        #     - (Aw_w.weak_form() - 1/2 * I.weak_form()) * self.wave.coefficients(large_cube[DOMAIN_OP])
        # ]

        # flatten list of lists and return
        return [y for x in rhs for y in x]

    def get_identity_op(self, space, domain, range_, dtr):
        """
        Shortcut to the sparse identity operator.
        """
        return bempp.api.operators.boundary.sparse.identity(
            # this can make the kernel crash if not set correctly
            space[domain], space[range_], space[dtr]
        )
    
    def manually_get_block_identity_op(self, space):
        """
        Create an identity operator matching the given space.
        """
        i = self.get_identity_op(space, 'RWG', 'RWG', 'SNC')
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


def get_simple_block_op(base, target, k, mu):
    """
    Return a 2x2 block operator defining the block matrix that would
    act on the given grid.
    
    This is similar to the `multitrace_operator` constructor, but it
    allows us to specify exactly which boundary disretisation functions
    to use.
    """
    efie = maxwell.electric_field(
        base['RWG'], target['RWG'], target['SNC'], k,
    )
    mfie = maxwell.magnetic_field(
        base['RWG'], target['RWG'], target['SNC'], k,
    )
    A = to_block_op(mfie, efie, k, mu)
    return A


# def get_mixed_block_op(target, source, k, mu, domain, range_, dtr):
#     """
#     Return a 2x2 block operator that defines the interferences on
#     `grid_a` by `grid_b`.
#     """
#     efie = maxwell.electric_field(
#         source[domain], target[range_], target[dtr], k
#     )
#     mfie = maxwell.magnetic_field(
#         source[domain], target[range_], target[dtr], k
#     )    
#     A = to_block_op(mfie, efie, k, mu)
#     return A


def assign_in_place_subblock(A, a, i, j):
    """
    Assigns the 4 elements of a to the 2x2 block of A
    specified by the indexes i and j.
    """
    bi = 2*i
    bj = 2*j
    A[bi,   bj]   = a[0, 0]
    A[bi,   bj+1] = a[0, 1]
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
    
    def show_domains(self):
        """
        todo
        """
        show_domains(self)

    def get_total_memory_size(self):
        """
        todo
        """
        memory = 0
        for i in range(4):
            for j in range(4):
                memory += self.system.operator[i, j].memory
        return memory
