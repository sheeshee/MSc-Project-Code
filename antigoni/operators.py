import bempp.api 
import numpy as np
from scipy.sparse import coo_matrix

def rescale(A, d1, d2):
    """Rescale the 2x2 block operator matrix A"""
    
    A[0, 1] = A[0, 1] * (d2 / d1)
    A[1, 0] = A[1, 0] * (d1 / d2)
    
    return A


def PMCHWT_operator(grids, k_ext, k_int, mu_ext, mu_int, block_discretisation = False, preconditioner = False, 
                    bary = False, parameters = None, sparse = False, type_of_preconditioner = False):
    """Set up the PMCHWT operator """
    
    if preconditioner == False and type_of_preconditioner != False:
        raise ValueError("Type of preconditioner should only be defined when preconditioner == True")
    if preconditioner == True and type_of_preconditioner == False:
        type_of_preconditioner = 'full'
    number_of_scatterers = len(grids)
    interior_operators = []
    exterior_operators = []
    identity_operators = []
    interior_electric_operators = []
    exterior_electric_operators = []
    mixed_electric_operators = []
    
    if sparse == True and preconditioner == False:
        raise ValueError('Sparse format currently implemented for preconditioners only')
        
    #setting up spaces. If preconditioner is true then we use BC and RBC functions. 
    #If preconditioner is false: 1. If bary is true we use RWG and SNC functions on the barycentric grid
    #2. If bary is false we use RWG and SNC functions on the original grid
    if preconditioner == True:
        domain_space = [bempp.api.function_space(grid, "BC", 0) for grid in grids]
        range_space = [bempp.api.function_space(grid, "BC", 0) for grid in grids]
        dual_to_range_space = [bempp.api.function_space(grid, "RBC", 0) for grid in grids]
    else:
        if bary == True:
            domain_space = [bempp.api.function_space(grid.barycentric_grid(), "RWG", 0) for grid in grids]
            range_space = [bempp.api.function_space(grid.barycentric_grid(), "RWG", 0) for grid in grids]
            dual_to_range_space = [bempp.api.function_space(grid.barycentric_grid(), "SNC", 0) for grid in grids]
        else:
            domain_space = [bempp.api.function_space(grid, "RWG", 0) for grid in grids]
            range_space = [bempp.api.function_space(grid, "RWG", 0) for grid in grids]
            dual_to_range_space = [bempp.api.function_space(grid, "SNC", 0) for grid in grids]
            
            
    for i in range(number_of_scatterers):
        if block_discretisation == True:
            A_ext = bempp.api.operators.boundary.maxwell.multitrace_operator(grids[i], k_ext, parameters = parameters)
            A_int = bempp.api.operators.boundary.maxwell.multitrace_operator(grids[i], k_int[i], parameters = parameters)
            
            ident = bempp.api.operators.boundary.sparse.multitrace_identity(grids[i], spaces='maxwell')
                                
        else:
            A_ext = bempp.api.assembly.BlockedOperator(2,2)
            A_int = bempp.api.assembly.BlockedOperator(2,2)

            magnetic_field_ext = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space[i], range_space[i],
                                                                                dual_to_range_space[i], k_ext,
                                                                                parameters = parameters)
            electric_field_ext = bempp.api.operators.boundary.maxwell.electric_field(domain_space[i], range_space[i], 
                                                                                dual_to_range_space[i], k_ext,
                                                                                parameters = parameters)
            A_ext[0,0] = magnetic_field_ext
            A_ext[0,1] = electric_field_ext
            A_ext[1,0] = -1 * electric_field_ext
            A_ext[1,1] = magnetic_field_ext

            magnetic_field_int = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space[i], range_space[i],
                                                                                dual_to_range_space[i], k_int[i],
                                                                                parameters = parameters)
            electric_field_int = bempp.api.operators.boundary.maxwell.electric_field(domain_space[i], range_space[i], 
                                                                                dual_to_range_space[i], k_int[i],
                                                                                parameters = parameters)
            A_int[0,0] = magnetic_field_int
            A_int[0,1] = electric_field_int
            A_int[1,0] = -1 * electric_field_int
            A_int[1,1] = magnetic_field_int
            
            ident = bempp.api.assembly.BlockedOperator(2,2)
            identity = bempp.api.operators.boundary.sparse.identity(domain_space[i], range_space[i], dual_to_range_space[i])
            ident[0,0] = identity
            ident[1,1] = identity
        
        E_int = bempp.api.assembly.BlockedOperator(2,2)
        E_ext = bempp.api.assembly.BlockedOperator(2,2)
        E_mixed = bempp.api.assembly.BlockedOperator(2,2)
        
        E_mixed[0,1] = (mu_int[i]/k_int[i]) * A_int[0,1]
        E_mixed[1,0] = (k_ext/mu_ext) * A_ext[1,0]
        
        A_ext = rescale(A_ext, k_ext, mu_ext)
        A_int = rescale(A_int, k_int[i], mu_int[i])
        
        E_int[0,1] = A_int[0,1]
        E_int[1,0] = A_int[1,0]
        
        E_ext[0,1] = A_ext[0,1]
        E_ext[1,0] = A_ext[1,0]
        
        interior_operators.append(A_int)
        exterior_operators.append(A_ext)
        identity_operators.append(ident)
        interior_electric_operators.append(E_int)
        exterior_electric_operators.append(E_ext)
        mixed_electric_operators.append(E_mixed)
        
    filter_operators = number_of_scatterers * [None]
    transfer_operators = np.empty((number_of_scatterers, number_of_scatterers), dtype=np.object)

    PMCHWT_op = bempp.api.BlockedOperator(2 * number_of_scatterers, 2 * number_of_scatterers)

    for i in range(number_of_scatterers):
        filter_operators[i] = .5 * identity_operators[i]- interior_operators[i]
        for j in range(number_of_scatterers):
            if i == j:
                # Create the diagonal elements
                if preconditioner == False:
                    element = interior_operators[j] + exterior_operators[j]
                elif type_of_preconditioner == 'full':
                    element = interior_operators[j] + exterior_operators[j]
                elif type_of_preconditioner == 'exterior':
                    element = exterior_operators[j]
                elif type_of_preconditioner == 'interior':
                    element = interior_operators[j]
                elif type_of_preconditioner == 'interior_electric':
                    element = interior_electric_operators[j]
                elif type_of_preconditioner == 'exterior_electric':
                    element = exterior_electric_operators[j]
                elif type_of_preconditioner == 'mixed':
                    element = mixed_electric_operators[j]
                else:
                    raise ValueError('Type of preconditioner can be False, full, exterior, interior, interior_electric, exterior_electric and mixed')
            else:
                # Do the off-diagonal elements
                if block_discretisation == True:
                    Aij = bempp.api.operators.boundary.maxwell.multitrace_operator(grids[j], k_ext, target=grids[i], 
                                                                                   parameters = parameters)
                else:
                    Aij = bempp.api.BlockedOperator(2,2) 
                    magnetic_field = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space[j], range_space[i], 
                                                                                         dual_to_range_space[i], k_ext, 
                                                                                         parameters=parameters) 
                    electric_field = bempp.api.operators.boundary.maxwell.electric_field(domain_space[j], range_space[i], 
                                                                                         dual_to_range_space[i], k_ext, 
                                                                                         parameters=parameters) 
                    
                    Aij[0,0] = magnetic_field
                    Aij[0,1] = electric_field
                    Aij[1,0] = -1 * electric_field
                    Aij[1,1] = magnetic_field

                transfer_operators[i, j] = rescale(Aij, k_ext, mu_ext)
                element= transfer_operators[i, j]

            #Assign the 2x2 element to the block operator matrix.
            PMCHWT_op[2 * i, 2 * j] = element[0, 0]
            PMCHWT_op[2 * i, 2 * j + 1] = element[0, 1]
            PMCHWT_op[2 * i + 1, 2 * j] = element[1, 0]
            PMCHWT_op[2 * i + 1, 2 * j + 1] = element[1, 1] 
    
    
    
    if preconditioner == True:
        pre_diag = bempp.api.BlockedOperator(2*number_of_scatterers, 2*number_of_scatterers)
    
        #set diagonal preconditioner
        for i in range(number_of_scatterers):
            pre_diag[2*i, 2*i] = PMCHWT_op[2*i, 2*i]
            pre_diag[2*i, 2*i+1] = PMCHWT_op[2*i, 2*i+1]
            pre_diag[2*i+1, 2*i] = PMCHWT_op[2*i+1, 2*i]
            pre_diag[2*i+1, 2*i+1] = PMCHWT_op[2*i+1, 2*i+1]
        return [PMCHWT_op, pre_diag]
    
    return [PMCHWT_op, filter_operators]

def multitrace_operator(grids, k, preconditioner = False, block_discretisation = False, bary = False, parameters = None, sparse = False, singular_assembly = False, diag_preconditioner = False):
    """Set up the multitrace operator for single-particle problem """
    
    grid = grids[0]
    if sparse == True and preconditioner == False:
        raise ValueError('Sparse format currently implemented for preconditioners only')
    if preconditioner == False and diag_preconditioner != False:
        raise ValueError("Can't have diagonal preconditioner without a preconditioner!")
        
    #setting up spaces. If preconditioner is true then we use BC and RBC functions. 
    #If preconditioner is false: 1. If bary is true we use RWG and SNC functions on the barycentric grid
    #2. If bary is false we use RWG and SNC functions on the original grid
    if preconditioner == True:
        domain_space = bempp.api.function_space(grid, "BC", 0)
        range_space = bempp.api.function_space(grid, "BC", 0)
        dual_to_range_space = bempp.api.function_space(grid, "RBC", 0) 
    else:
        if bary == True:
            domain_space = bempp.api.function_space(grid.barycentric_grid(), "RWG", 0)
            range_space = bempp.api.function_space(grid.barycentric_grid(), "RWG", 0) 
            dual_to_range_space = bempp.api.function_space(grid.barycentric_grid(), "SNC", 0)
        else:
            domain_space = bempp.api.function_space(grid, "RWG", 0)
            range_space = bempp.api.function_space(grid, "RWG", 0) 
            dual_to_range_space = bempp.api.function_space(grid, "SNC", 0) 
            
            
    
    if block_discretisation == True:
        A = bempp.api.operators.boundary.maxwell.multitrace_operator(grid, k, parameters = parameters)

        ident = bempp.api.operators.boundary.sparse.multitrace_identity(grid, spaces='maxwell')
    else:
        A = bempp.api.assembly.BlockedOperator(2,2)

        magnetic_field = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space, range_space,
                                                                             dual_to_range_space, k,
                                                                             parameters = parameters,
                                                                             assemble_only_singular_part = singular_assembly)
        electric_field = bempp.api.operators.boundary.maxwell.electric_field(domain_space, range_space,
                                                                             dual_to_range_space, k,
                                                                             parameters = parameters,
                                                                             assemble_only_singular_part = singular_assembly)
        if diag_preconditioner == True:
            A[0,1] = electric_field
            A[1,0] = -1 * electric_field
        else:
            A[0,0] = magnetic_field
            A[0,1] = electric_field
            A[1,0] = -1 * electric_field
            A[1,1] = magnetic_field


        ident = bempp.api.assembly.BlockedOperator(2,2)
        identity = bempp.api.operators.boundary.sparse.identity(domain_space, range_space, dual_to_range_space)
        ident[0,0] = identity
        ident[1,1] = identity
        
        
    return [A, ident]

def convert_to_sparse(op, grid):
    """ Stores the h-matrix into SparseDiscreteOperator coo_matrix """
    t0 = time.time()
    op_wf = op.weak_form()
    ta_wf = time.time() - t0
    set_of_operators = op_wf.elementary_operators()
    bc_space = bempp.api.function_space(grid, "BC", 0)
    rbc_space = bempp.api.function_space(grid, "RBC", 0)
    b_rwg_space = bempp.api.function_space(grid.barycentric_grid(), "RWG", 0)
    b_snc_space = bempp.api.function_space(grid.barycentric_grid(), "SNC", 0)
    
    for operator in set_of_operators:
        if type(operator) == bempp.api.assembly.discrete_boundary_operator.GeneralNonlocalDiscreteBoundaryOperator:
            tree = bempp.api.hmatrix_interface.block_cluster_tree(operator)
            (rows_hmat_to_original_dofs, columns_hmat_to_original_dofs) = tree.hmat_dofs_to_original_dofs
            root = tree.root
            inadmissible_nodes = [node for node in tree.leaf_nodes if not node.admissible]
            admissible_nodes = [node for node in tree.leaf_nodes if node.admissible]    
            
            data = np.zeros(operator.shape, dtype='complex128')
            
            for item in admissible_nodes:
                rows_range = item.row_cluster_range
                cols_range = item.column_cluster_range
                data_block = bempp.api.hmatrix_interface.data_block(operator, item)
                item_matrixA = data_block.A
                item_matrixB  = data_block.B
                item_matrix = item_matrixA.dot(item_matrixB)
                row_index = 0
                for i in range(rows_range[0], rows_range[1]):
                    column_index = 0
                    for j in range(cols_range[0], cols_range[1]):
                        data[i,j] = item_matrix[row_index, column_index]
                        column_index += 1
                    row_index += 1
        
            for item in inadmissible_nodes:
                rows_range = item.row_cluster_range
                cols_range = item.column_cluster_range
                data_block = bempp.api.hmatrix_interface.data_block(operator, item)
                item_matrix = data_block.A

                row_index = 0
                for i in range(rows_range[0], rows_range[1]):
                    column_index = 0
                    for j in range(cols_range[0], cols_range[1]):
                        data[i,j] = item_matrix[row_index, column_index]
                        column_index += 1
                    row_index += 1

            data_hmat_to_original = np.zeros(operator.shape, dtype = 'complex128')
        
            for i in range(0, operator.shape[0]):
                row_original_dof = rows_hmat_to_original_dofs[i]
                for j in range(0, operator.shape[1]):
                    col_original_dof = columns_hmat_to_original_dofs[j]
                    data_hmat_to_original[row_original_dof, col_original_dof] = data[i,j]
            
          
            coo = coo_matrix(data_hmat_to_original)
            sparse_op = bempp.api.assembly.discrete_boundary_operator.SparseDiscreteBoundaryOperator(coo)

            ident1 = bempp.api.operators.boundary.sparse.identity(b_snc_space, rbc_space, rbc_space)
            ident2 = bempp.api.operators.boundary.sparse.identity(bc_space, b_rwg_space, b_rwg_space)
            ident1_wf = ident1.weak_form()
            ident2_wf = ident2.weak_form()
            inv1 = bempp.api.operators.boundary.sparse.identity(b_snc_space, b_snc_space, b_snc_space)
            inv1_wf = inv1.weak_form()
            inv1_sparse = bempp.api.assembly.discrete_boundary_operator.InverseSparseDiscreteBoundaryOperator(inv1_wf)
            inv2 = bempp.api.operators.boundary.sparse.identity(b_rwg_space, b_rwg_space, b_rwg_space)
            inv2_wf = inv2.weak_form()
            inv2_sparse = bempp.api.assembly.discrete_boundary_operator.InverseSparseDiscreteBoundaryOperator(inv2_wf)
            
            result = ident1_wf * inv1_sparse * sparse_op * inv2_sparse * ident2_wf
            
#             op1_wf_matrix = bempp.api.as_matrix(op_wf)
#             plt.spy(op1_wf_matrix, precision = 1E-10)
#             plt.show()
            
#             sparse_op_matrix = bempp.api.as_matrix(result)
#             plt.spy(sparse_op_matrix, precision = 1E-10)
#             plt.show()
            
#             print(len([i for i in (op1_wf_matrix - sparse_op_matrix).flatten() if np.absolute(i)>1E-10]))
    return result, ta_wf