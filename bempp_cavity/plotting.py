"""
todo
"""
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

import bempp.api
from bempp.api.operators.potential import maxwell as maxwell_potential


class Plotter:
    def __init__(self):
        """
        todo
        """
        pass


class StrongFromPlotter(Plotter):
    def __init__(self):
        """
        todo
        """
        pass


class WeakFormPlotter(Plotter):
    def __init__(self):
        """
        todo
        """
        pass


Vector = namedtuple('Vector', ['dirichlet', 'neumann'])


def weak_form_plot(solution):
    DOMAIN_OP = 'RWG'
    number_of_points = 220
    domains = [cavity[DOMAIN_OP] for cavity in solution.system.cavities] \
        + [solution.system.main[DOMAIN_OP]]
    points, cavity_indexers, wall_indexer, exterior_indexer, limits, shape = setup_plotting(
        number_of_points, solution)
   
    u_cavities, u_scattered = partition_data(solution, domains)    
    u_wall = Vector(
        dirichlet=u_scattered.dirichlet + solution.system.wave.dirichlet_trace(solution.system.main['RWG']).coefficients,
        neumann=u_scattered.neumann + solution.system.wave.neumann_trace(solution.system.main['RWG']).coefficients
    )

    trace_cavities = [
        to_trace(space['RWG'], u) for space, u in zip(solution.system.cavities, u_cavities)
    ]
    trace_wall = to_trace(solution.system.main, u_wall)

    cavity_fields = calculate_cavity_fields_weak(solution, points, cavity_indexers, domains, trace_cavities)
    wall_field = calculate_wall_field_weak(solution, points, wall_indexer, domains, trace_wall, trace_cavities)
    scattered_field = calculate_scattered_field_weak(solution, points, exterior_indexer, domains, trace_wall)
    
    # # plot
    total_field = np.empty_like(points, dtype='complex128')
    for indexer, field in zip(cavity_indexers, cavity_fields):
        total_field[:, indexer] = field
    total_field[:, wall_indexer] = wall_field
    total_field[:, exterior_indexer] = scattered_field + solution.system.wave.incident_field(points[:, exterior_indexer])
    squared_field = np.sum(np.abs(total_field**2), axis=0)
    implot(limits, shape, squared_field)
    plt.show()


def calculate_scattered_field_weak(solution, points, exterior_indexer, domains, trace_wall):
    """
    todo
    """
    exterior_points = points[:, exterior_indexer]
    H_pot = maxwell_potential.magnetic_field(
        domains[-1], exterior_points, solution.system.wave_numbers[0]
    )
    E_pot = maxwell_potential.electric_field(
        domains[-1], exterior_points, solution.system.wave_numbers[0]
    )
    k = solution.system.wave_numbers[0]
    mu = solution.system.mu_numbers[0]
    scattered_field = - H_pot * trace_wall.dirichlet - E_pot * (mu/k * trace_wall.neumann)
    return scattered_field


def calculate_wall_field_weak(solution, points, wall_indexer, domains, trace_wall, trace_cavities):
    """
    todo
    """
    # wall
    wall_points = points[:, wall_indexer]
    kw = solution.system.wave_numbers[-1]
    mw = solution.system.wave_numbers[-1]
    H_pot_w = maxwell_potential.magnetic_field(domains[-1], wall_points, kw)
    E_pot_w = maxwell_potential.electric_field(domains[-1], wall_points, kw)

    H_pot_c = []
    E_pot_c = []
    kc_list = []
    mu_list = []
    for i in range(solution.system.N-1):
        kc = solution.system.wave_numbers[i+2]
        kc_list.append(kc)
        mu_list.append(solution.system.mu_numbers[i+2])
        H_pot_c.append(
            maxwell_potential.magnetic_field(
                domains[i], wall_points, kc
            )
        )
        E_pot_c.append(
            maxwell_potential.electric_field(
                domains[i], wall_points, kc
            )
        )
    
    wall_field = H_pot_w * trace_wall.dirichlet + E_pot_w * (mw/kw * trace_wall.neumann) \
        + sum([
            H_pot_c[i] * trace_cavities[i].dirichlet \
                + E_pot_c[i] * (mu_list[i]/kc_list[i] * trace_cavities[i].neumann)
        ])
    return wall_field


def calculate_cavity_fields_weak(solution, points, cavity_indexers, domains, traces):
    """
    todo
    """
    E_pot_cavities = []
    H_pot_cavities = []
    # print(cavity_indexers)
    for i in range(solution.system.N-1):
        cavity_points = points[:, cavity_indexers[i]] # [:, :, 0]
        H_pot_cavities.append(
            maxwell_potential.magnetic_field(
                domains[i], cavity_points, solution.system.wave_numbers[i+2]
            )
        )
        E_pot_cavities.append(
            maxwell_potential.electric_field(
                domains[i], cavity_points, solution.system.wave_numbers[i+2]
            )
        )
    ## cavities
    cavity_fields = [
        H_pot_cavities[i] * trace.dirichlet \
            + E_pot_cavities[i] * \
                (solution.system.mu_numbers[2+i]/solution.system.wave_numbers[2+i] * \
                    trace.neumann)
        for i, trace in enumerate(traces)
    ]
    return cavity_fields
    

def partition_data(solution, domains):
    """
    todo
    """
    x = solution.coefficients
    u_cavities = []
    for i in range(solution.system.N-1):
        dStart = 2*i*domains[i].global_dof_count
        dEnd   = dStart + domains[i].global_dof_count
        nStart = dEnd
        nEnd   = dEnd+domains[i].global_dof_count
        u_cavities.append(
            Vector(
                dirichlet=x[dStart:dEnd],
                neumann=x[nStart:nEnd]
            )
        )

    u_scattered = Vector(
        dirichlet=x[nEnd:nEnd+domains[-1].global_dof_count],
        neumann=x[nEnd+domains[-1].global_dof_count:]
    )
    return u_cavities, u_scattered


def to_trace(space, u):
    """
    todo
    """
    return Vector(
        dirichlet=bempp.api.GridFunction(
            space, coefficients=u.dirichlet),
        neumann=bempp.api.GridFunction(
            space, coefficients=u.neumann)
    )


def weak_form_plot_old(solution):
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
    domains = solution.system.operator.domain_spaces
    number_of_points = 220
    points, cavity_indexers, wall_indexer, exterior_indexer, limits, shape = setup_plotting(
        number_of_points, solution)
    
    wall_dTrace = solution.system.wave.dirichlet_trace(
        solution.system.operator.domain_spaces[-2]) + solution.traces[-2]
    wall_nTrace = solution.system.wave.dirichlet_trace(
        solution.system.operator.domain_spaces[-1]) + solution.traces[-1]
    
    cavity_fields = calculate_cavity_fields(solution, points, cavity_indexers, domains)
    wall_field = calculate_wall_field(solution, points, wall_indexer, domains, wall_dTrace, wall_nTrace)
    scattered_field = calculate_scattered_field(points, exterior_indexer, domains, solution)

    # plot
    total_field = np.empty_like(points, dtype='complex128')
    for indexer, field in zip(cavity_indexers, cavity_fields):
        total_field[:, indexer] = field
    total_field[:, wall_indexer] = wall_field
    total_field[:, exterior_indexer] = scattered_field + solution.system.wave.incident_field(points[:, exterior_indexer])
    squared_field = np.sum(np.abs(total_field**2), axis=0)
    implot(limits, shape, squared_field)
    plt.show()


def setup_plotting(number_of_points, solution):
    shape = (number_of_points, number_of_points)
    limits = get_limits(solution.system.cavity_grid.main.bounding_box, 0.5)
    points = get_point_cloud(limits, number_of_points)
    exterior_indexer, wall_indexer, cavity_indexers = get_indexers(points, solution)
    return points, cavity_indexers, wall_indexer, exterior_indexer, limits, shape


def calculate_scattered_field(points, exterior_indexer, domains, solution):
    """
    todo
    """
    exterior_points = points[:, exterior_indexer] # [:, :, 0]
    H_pot_ext = maxwell_potential.magnetic_field(
        domains[-2], exterior_points, solution.system.wave_numbers[0]
    )
    E_pot_ext = maxwell_potential.magnetic_field(
        domains[-1], exterior_points, solution.system.wave_numbers[0]
    )
    scattered_field = - H_pot_ext * solution.traces[-2] \
        - E_pot_ext * (
            solution.system.mu_numbers[0]/solution.system.wave_numbers[0] * solution.traces[-1]
        )
    return scattered_field


def calculate_wall_field(solution, points, wall_indexer, domains, wall_dTrace, wall_nTrace):
    # wall
    wall_points = points[:, wall_indexer]
    H_pot_wall = maxwell_potential.magnetic_field(
        domains[-2], wall_points, solution.system.wave_numbers[1])
    E_pot_wall = maxwell_potential.electric_field(
        domains[-1], wall_points, solution.system.wave_numbers[1])

    H_pot_cavities = []
    E_pot_cavities = []
    
    # Influence of cavity
    for i in range(solution.system.N-1):
        H_pot_cavities.append(
            maxwell_potential.magnetic_field(
                domains[2*i], wall_points, solution.system.wave_numbers[1])
        )
        E_pot_cavities.append(
            maxwell_potential.electric_field(
                domains[2*i+1], wall_points, solution.system.wave_numbers[1])
        )

    cavity_contribution = sum([
        H_pot_cavities[i] * solution.traces[2*i] \
            + E_pot_cavities[i] * (
                solution.system.mu_numbers[2+i]/solution.system.wave_numbers[2+i] \
                    * solution.traces[2*i+1]
            )
        for i in range(solution.system.N-1)
    ])
    
    wall_field = H_pot_wall * wall_dTrace + E_pot_wall * (solution.system.mu_numbers[1]/solution.system.wave_numbers[1] \
        * wall_nTrace) - cavity_contribution
    return wall_field


def calculate_cavity_fields(solution, points, cavity_indexers, domains):
    # cavities
    E_pot_cavities = []
    H_pot_cavities = []
    # print(cavity_indexers)
    for i in range(solution.system.N-1):
        
        cavity_points = points[:, cavity_indexers[i]] # [:, :, 0]
        H_pot_cavities.append(
            maxwell_potential.magnetic_field(
                domains[2*i], cavity_points, solution.system.wave_numbers[i+2]
            )
        )
        E_pot_cavities.append(
            maxwell_potential.electric_field(
                domains[2*i+1], cavity_points, solution.system.wave_numbers[i+2]
            )
        )
    # fields
    ## cavities
    cavity_fields = [
        H_pot_cavities[i] * solution.traces[2*i] \
            + E_pot_cavities[i] * \
                (solution.system.mu_numbers[2+i]/solution.system.wave_numbers[2+i] * \
                    solution.traces[2*i+1])
        for i, _ in enumerate(solution.system.cavity_grid.cavities)
    ]
    return cavity_fields

  
def show_domains(solution):
    """
    todo
    """
    number_of_points = 220
    points, cavity_indexers, wall_indexer, exterior_indexer, limits, shape = setup_plotting(number_of_points, solution)
    
    regions = np.empty_like(points)
    
    regions[:, exterior_indexer] = 1
    regions[:, wall_indexer] = 1.5
    for indexer in cavity_indexers:
        regions[:, indexer] = 0

    data = np.sum(np.abs(regions**2), axis=0)
    implot(limits, shape, data, clim=(0, 20))


def get_indexers(points, solution):
    """
    todo
    """
    # check if the point is exterior to outer boundary
    exterior_indexer = get_indices(points, solution.system.cavity_grid.main, -1)
    # check if the point is within any of the cavities
    cavity_indexers = [
        get_indices(points, cavity)
        for cavity in solution.system.cavity_grid.cavities
    ]
    # else, conclude the point is in the wall
    mask = np.ones(len(points.T), np.bool)
    mask[exterior_indexer] = 0
    for indexer in cavity_indexers:
        mask[indexer] = 0
    wall_indexer = np.arange(len(points.T))[mask]
    return exterior_indexer, wall_indexer, cavity_indexers

    
# def strong_form_plot_old(solution):
#     """
#     todo
#     """
#     shape = (220, 220)

#     KE = solution.system.wave_numbers[0]
#     KW = solution.system.wave_numbers[1]
#     KI = solution.system.wave_numbers[2]
#     MU = solution.system.mu_numbers[0]
#     print(KE, KW, KI, MU)

#     small_domain_0 = solution.system.operator.domain_spaces[0]
#     small_domain_1 = solution.system.operator.domain_spaces[1]
#     large_domain_0 = solution.system.operator.domain_spaces[2]
#     large_domain_1 = solution.system.operator.domain_spaces[3]
    
#     # cavity
#     Dtrace_i = solution.traces[0]
#     Ntrace_i = solution.traces[1]
    
#     # scattered
#     Dtrace = solution.traces[2]
#     Ntrace = solution.traces[3]
    
#     # wall
#     Dtrace_w = solution.system.wave.dirichlet_trace(solution.system.operator.domain_spaces[2]) + Dtrace
#     Ntrace_w = solution.system.wave.neumann_trace(solution.system.operator.domain_spaces[3]) + Ntrace

#     points, limits, cavity_indexer, wall_indexer, exterior_indexer = get_spaces(shape, 2, 1)
#     print([len(i) for i in [cavity_indexer, wall_indexer, exterior_indexer]])

#     # Cavity
#     print('cavity')
#     cavity_points = points[:, cavity_indexer]

#     E_potential_op = maxwell_potential.electric_field(
#         small_domain_1, cavity_points, KI)
#     H_potential_op = maxwell_potential.magnetic_field(
#         small_domain_0, cavity_points, KI)

#     cavity_field =  H_potential_op * Dtrace_i + E_potential_op * (MU/KI * Ntrace_i)

#     # Wall
#     print('wall')
#     wall_points = points[:, wall_indexer]

#     # Influence of external boundary
#     E_potential_op_w = maxwell_potential.electric_field(
#         large_domain_1, wall_points, KW)
#     H_potential_op_w = maxwell_potential.magnetic_field(
#         large_domain_0, wall_points, KW)

#     # Influence of cavity
#     E_potential_op_i = maxwell_potential.electric_field(
#         small_domain_1, wall_points, KW)
#     H_potential_op_i = maxwell_potential.magnetic_field(
#         small_domain_0, wall_points, KW)

#     # Putting them together
#     wall_field = H_potential_op_w * Dtrace_w + E_potential_op_w * (MU/KW * Ntrace_w) \
#         - (H_potential_op_i * Dtrace_i + E_potential_op_i * (MU/KI * Ntrace_i))

#     # Scattered
#     print('exterior')
#     exterior_points = points[:, exterior_indexer]

#     E_potential_op = maxwell_potential.electric_field(
#         large_domain_1, exterior_points, KE)
#     H_potential_op = maxwell_potential.magnetic_field(
#         large_domain_0, exterior_points, KE)

#     scattered_field = - H_potential_op * Dtrace - E_potential_op * (MU/KE * Ntrace)

#     # plot
#     total_field = np.empty_like(points, dtype='complex128')
#     total_field[:, cavity_indexer] = cavity_field
#     total_field[:, wall_indexer] = wall_field
#     total_field[:, exterior_indexer] = scattered_field + solution.system.wave.incident_field(points[:, exterior_indexer])
#     squared_field = np.sum(np.abs(total_field**2), axis=0)
#     implot(limits, shape, squared_field)
#     plt.show()


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


def get_point_cloud(limits, number_of_points):
    """
    todo
    """
    xmin, xmax, zmin, zmax = limits
    plot_grid = np.mgrid[
        xmin:xmax:number_of_points * 1j,
        0:0:1j, # sectional splice
        zmin:zmax:number_of_points * 1j
    ]

    c = 0 # height of intersecting plane

    points = np.vstack((
        plot_grid[0].ravel(),
        c * np.ones(plot_grid[0].size),
        plot_grid[2].ravel()
    ))

    return points

    
def get_limits(bounding_box, padding):
    """
    todo

    array([[-1.5, -1.5, -1.5],
            [ 1.5,  1.5,  1.5]])
    """
    xmin = bounding_box[0, 0] - padding
    xmax = bounding_box[1, 0] + padding
    zmin = bounding_box[0, 2] - padding
    zmax = bounding_box[1, 2] + padding
    return xmin, xmax, zmin, zmax


def get_indices(points, grid, direction=1):
    """
    todo
    """
    box = grid.bounding_box
    if direction == -1:
    
        x = (points[0, :] < box[0, 0]) | (box[1, 0] < points[0, :])
        y = (points[1, :] < box[0, 1]) | (box[1, 1] < points[1, :])
        z = (points[2, :] < box[0, 2]) | (box[1, 2] < points[2, :])        
        to_return =  np.argwhere(x | y | z) # must include y as well
    else:
        x = (box[0, 0] < points[0, :]) * (points[0, :] < box[1, 0])
        y = (box[0, 1] < points[1, :]) * (points[1, :] < box[1, 1])
        z = (box[0, 2] < points[2, :]) * (points[2, :] < box[1, 2])
        
        to_return =  np.argwhere(x & y & z) # must include y as well
    return np.ravel(to_return)
