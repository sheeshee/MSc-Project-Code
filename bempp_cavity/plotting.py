"""
todo
"""
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

import bempp.api
from bempp.api.operators.potential import maxwell as maxwell_potential



Vector = namedtuple('Vector', ['dirichlet', 'neumann'])


def weak_form_plot(solution):
    """
    todo
    """
    shape = (220, 220)
    DOMAIN_OP = 'RWG'

    KE = solution.system.wave_numbers[0]
    KW = solution.system.wave_numbers[1]
    KI = solution.system.wave_numbers[2]
    MU = solution.system.mu_numbers[0]
    print(KE, KW, KI, MU)

    small_domain_0 = solution.system.cavities[0][DOMAIN_OP]
    small_domain_1 = solution.system.cavities[0][DOMAIN_OP]
    large_domain_0 = solution.system.main[DOMAIN_OP]
    large_domain_1 = solution.system.main[DOMAIN_OP]

    # Split between different domains in vector
    x = solution.coefficients
    u_cavity = Vector(
        dirichlet=x[:small_domain_0.global_dof_count],
        neumann=x[small_domain_0.global_dof_count:small_domain_0.global_dof_count*2]
    )

    split_point = small_domain_0.global_dof_count*2

    u_scattered = Vector(
        dirichlet=x[split_point:split_point+large_domain_1.global_dof_count],
        neumann=x[split_point+large_domain_1.global_dof_count:]
    )

    # Check
    def assert_lengths_match(a, b):
        """
        Raise Error if lengths of a and b do not match.
        """
        if isinstance(a, int):
            va = a
        else:
            va = len(a)

        if isinstance(b, int):
            vb = b
        else:
            vb = len(b)

        assert va == vb, "Lengths must match. Got %i and %i." % (va, vb)


    assert_lengths_match(
        sum([len(z) for z in [
            u_cavity.neumann, u_cavity.dirichlet,
            u_scattered.neumann, u_scattered.dirichlet
        ]]),
        x
    )

    assert_lengths_match(*u_cavity)
    assert_lengths_match(*u_scattered)


    u_wall = Vector(
        solution.system.wave.dirichlet_trace(large_domain_0).coefficients \
            + u_scattered.dirichlet,
        solution.system.wave.neumann_trace(large_domain_0).coefficients \
             + u_scattered.neumann,
    )

    # Cavity
    Ntrace_i = bempp.api.GridFunction(
        small_domain_1, coefficients=u_cavity.neumann)
    Dtrace_i = bempp.api.GridFunction(
        small_domain_0, coefficients=u_cavity.dirichlet)

    # Wall
    Ntrace_w = bempp.api.GridFunction(
        large_domain_1, coefficients=u_wall.neumann)
    Dtrace_w = bempp.api.GridFunction(
        large_domain_0, coefficients=u_wall.dirichlet)

    # Scattered
    Ntrace = bempp.api.GridFunction(
        large_domain_1, coefficients=u_scattered.neumann)
    Dtrace = bempp.api.GridFunction(
        large_domain_0, coefficients=u_scattered.dirichlet)
    
    # get points
    points, limits, cavity_indexer, wall_indexer, exterior_indexer = get_spaces(shape, 2, 1)
    print([len(i) for i in [cavity_indexer, wall_indexer, exterior_indexer]])

    # Cavity
    print('cavity')
    cavity_points = points[:, cavity_indexer]

    E_potential_op = maxwell_potential.electric_field(
        small_domain_1, cavity_points, KI)
    H_potential_op = maxwell_potential.magnetic_field(
        small_domain_0, cavity_points, KI)

    cavity_field =  H_potential_op * Dtrace_i + E_potential_op * (MU/KI * Ntrace_i)

    # Wall
    print('wall')
    wall_points = points[:, wall_indexer]

    # Influence of external boundary
    E_potential_op_w = maxwell_potential.electric_field(
        large_domain_1, wall_points, KW)
    H_potential_op_w = maxwell_potential.magnetic_field(
        large_domain_0, wall_points, KW)

    # Influence of cavity
    E_potential_op_i = maxwell_potential.electric_field(
        small_domain_1, wall_points, KW)
    H_potential_op_i = maxwell_potential.magnetic_field(
        small_domain_0, wall_points, KW)

    # Putting them together
    wall_field = H_potential_op_w * Dtrace_w + E_potential_op_w * (MU/KW * Ntrace_w) \
        - (H_potential_op_i * Dtrace_i + E_potential_op_i * (MU/KI * Ntrace_i))

    # Scattered
    print('exterior')
    exterior_points = points[:, exterior_indexer]

    E_potential_op = maxwell_potential.electric_field(
        large_domain_1, exterior_points, KE)
    H_potential_op = maxwell_potential.magnetic_field(
        large_domain_0, exterior_points, KE)


    scattered_field = - H_potential_op * Dtrace - E_potential_op * (MU/KE * Ntrace)

    # plot
    total_field = np.empty_like(points, dtype='complex128')
    total_field[:, cavity_indexer] = cavity_field
    total_field[:, wall_indexer] = wall_field
    total_field[:, exterior_indexer] = scattered_field + solution.system.wave.incident_field(points[:, exterior_indexer])
    squared_field = np.sum(np.abs(total_field**2), axis=0)
    implot(limits, shape, squared_field)
    plt.show()


    
def strong_form_plot(solution):
    """
    todo
    """
    shape = (220, 220)

    KE = solution.system.wave_numbers[0]
    KW = solution.system.wave_numbers[1]
    KI = solution.system.wave_numbers[2]
    MU = solution.system.mu_numbers[0]
    print(KE, KW, KI, MU)

    small_domain_0 = solution.system.operator.domain_spaces[0]
    small_domain_1 = solution.system.operator.domain_spaces[1]
    large_domain_0 = solution.system.operator.domain_spaces[2]
    large_domain_1 = solution.system.operator.domain_spaces[3]
    
    # cavity
    Dtrace_i = solution.traces[0]
    Ntrace_i = solution.traces[1]
    
    # scattered
    Dtrace = solution.traces[2]
    Ntrace = solution.traces[3]
    
    # wall
    Dtrace_w = solution.system.wave.dirichlet_trace(solution.system.operator.domain_spaces[2]) + Dtrace
    Ntrace_w = solution.system.wave.neumann_trace(solution.system.operator.domain_spaces[3]) + Ntrace

    points, limits, cavity_indexer, wall_indexer, exterior_indexer = get_spaces(shape, 2, 1)
    print([len(i) for i in [cavity_indexer, wall_indexer, exterior_indexer]])

    # Cavity
    print('cavity')
    cavity_points = points[:, cavity_indexer]

    E_potential_op = maxwell_potential.electric_field(
        small_domain_1, cavity_points, KI)
    H_potential_op = maxwell_potential.magnetic_field(
        small_domain_0, cavity_points, KI)

    cavity_field =  H_potential_op * Dtrace_i + E_potential_op * (MU/KI * Ntrace_i)

    # Wall
    print('wall')
    wall_points = points[:, wall_indexer]

    # Influence of external boundary
    E_potential_op_w = maxwell_potential.electric_field(
        large_domain_1, wall_points, KW)
    H_potential_op_w = maxwell_potential.magnetic_field(
        large_domain_0, wall_points, KW)

    # Influence of cavity
    E_potential_op_i = maxwell_potential.electric_field(
        small_domain_1, wall_points, KW)
    H_potential_op_i = maxwell_potential.magnetic_field(
        small_domain_0, wall_points, KW)

    # Putting them together
    wall_field = H_potential_op_w * Dtrace_w + E_potential_op_w * (MU/KW * Ntrace_w) \
        - (H_potential_op_i * Dtrace_i + E_potential_op_i * (MU/KI * Ntrace_i))

    # Scattered
    print('exterior')
    exterior_points = points[:, exterior_indexer]

    E_potential_op = maxwell_potential.electric_field(
        large_domain_1, exterior_points, KE)
    H_potential_op = maxwell_potential.magnetic_field(
        large_domain_0, exterior_points, KE)


    scattered_field = - H_potential_op * Dtrace - E_potential_op * (MU/KE * Ntrace)

    # plot
    total_field = np.empty_like(points, dtype='complex128')
    total_field[:, cavity_indexer] = cavity_field
    total_field[:, wall_indexer] = wall_field
    total_field[:, exterior_indexer] = scattered_field + solution.system.wave.incident_field(points[:, exterior_indexer])
    squared_field = np.sum(np.abs(total_field**2), axis=0)
    implot(limits, shape, squared_field)
    plt.show()


def get_spaces(shape, LENGTH_WALL, LENGTH_CAVITY):
    """
    todo
    """
    # Number of points in the x-direction
    nx = shape[0] # 220# divide accordingly to achieve smaller particles

    # Number of points in the y-direction
    nz = shape[1] # 220

    xmin, xmax, zmin, zmax = [
        -LENGTH_WALL/2-1,
        LENGTH_WALL/2+1,
        -LENGTH_WALL/2-1,
        LENGTH_WALL/2+1
    ] 
    # Ask Antigoni, why j's
    plot_grid = np.mgrid[xmin:xmax:nx * 1j, 0:0:1j, zmin:zmax:nz * 1j]

    c = 0 # Intersecting plane

    points = np.vstack((
        plot_grid[0].ravel(),
        c * np.ones(plot_grid[0].size),
        plot_grid[2].ravel()
    ))

    cavity_indexer = []
    wall_indexer = []
    exterior_indexer = []

    for i, point in enumerate(points.T):
        if point_is_within_cube(LENGTH_CAVITY, point):
            cavity_indexer.append(i)
        elif point_is_within_cube(LENGTH_WALL, point):
            wall_indexer.append(i)
        elif point_is_within_cube(np.inf, point):
            exterior_indexer.append(i)
        else:
            raise ValueError("Point %s not within domain" % point)
    
    return points, (xmin, xmax, zmin, zmax), cavity_indexer, wall_indexer, exterior_indexer


def point_is_within_cube(upper_bound_length, point):
    """
    Determines whether the given point is contained within the a centered cube with the
    bounds specified.
    """
    c =  [-upper_bound_length/2 < px and px < upper_bound_length/2
          for px in point]
    return all(c)


def implot(limits, shape, data, clim=(0, 2)):
    """
    Plot on an axis the field data
    """
    im = plt.imshow(
        data.reshape(shape[0], shape[1]), # (nx, nz),
        extent=limits, # [xmin, xmax, zmin, zmax],
        clim=clim,
    )
    fig = plt.gcf()
    plt.colorbar()
    fig.dpi = 100


def get_field(points, selector, field):
    total_field = np.empty_like(points, dtype='complex128')
    total_field[:] = np.nan
    total_field[:, selector] = field
    return np.sum(np.abs(total_field**2), axis=0)
