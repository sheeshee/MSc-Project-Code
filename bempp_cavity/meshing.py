"""

"""


def create_grid(main_boundary, *nested_boundaries):
    """
    Creates a CavityGrid object from the given elements.
    """
    return CavityGrid(main_boundary, nested_boundaries)

class CavityGrid:
    """
    Contains the main grid, in which are nested all
    the cavity boundaries.
    """
    def __init__(self, main, cavities):
        self.main = main
        self.cavities = cavities
        # either create or load depending on args
