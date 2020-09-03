"""
This file contains a simple class that wraps the BEMPP parameters object
simplifying its definition.
"""

from bempp.core.common.global_parameters import global_parameters

class Parameters:
    """
    A simple interface to BEMPP's parameters object.
    Simplifies the definition of the parameters.
    """
    def __init__(self, **kwargs):
        """
        Load the global parameters object and specify its attributes as necessary.
        """
        self.nearfield_cutoff = kwargs['nearfield_cutoff']
        self.ACA = kwargs['ACA']
        if 'quadrature' in kwargs:
            assert(len(kwargs['quadrature']) == 4), \
                "Quadrature rule must have length 4."
            self.quadrature = kwargs['quadrature']

        self.params = global_parameters()
        # setup

        if self.nearfield_cutoff < 0:
            self.params.assembly.boundary_operator_assembly_type = 'dense'
        else:
            self.params.assembly.boundary_operator_assembly_type = 'hmat'
        
        self.params.hmat.eps = self.ACA
        self.params.hmat.cutoff = self.nearfield_cutoff
        self.params.quadrature.double_singular = self.quadrature[3]
        self.params.quadrature.far.double_order = self.quadrature[2]
        self.params.quadrature.medium.double_order = self.quadrature[1]
        self.params.quadrature.near.double_order = self.quadrature[0]

    def get(self):
        """
        Return the BEMPP parameters object.
        """
        return self.params
