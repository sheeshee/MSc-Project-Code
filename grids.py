"""
This file contains some constructor methods for objects with cavities.

Import these functions to quickly try different number of cube
cavities within the main boundary.

>>> import bempp_cavity
>>> import grids
>>> grid = make_grid_4()
>>> bempp_cavity.plotting.show_domains(grid)
"""
from bempp.api import shapes

import bempp_cavity

ELEMENT_SIZE = 0.5

def make_grid_1():
    """
    Returns a CavityGrid object of a solid with one cavity at the
    center.
    """
    inner_walls = []
    inner_walls.append(
        shapes.cube(
            length=2.5, h=ELEMENT_SIZE,
            origin=(-1.25, -1/2, -1.25)
        ),
    )
    outer_wall = shapes.cube(
        length=3,
        h=ELEMENT_SIZE, 
        origin=(-1.5, -1.5, -1.5)
    )
    return bempp_cavity.create_grid(outer_wall, *inner_walls)


def make_grid_4():
    """
    Returns a CavityGrid object of a solid with four cavities within it
    arranged in a square.
    """
    inner_walls = [
        shapes.cube(
            length=1.125, h=ELEMENT_SIZE,
            origin=(-1.25, -1/2, -1.25)
        ),
        shapes.cube(
            length=1.125, h=ELEMENT_SIZE,
            origin=(0.125, -1/2, -1.25)
        ),
        shapes.cube(
            length=1.125, h=ELEMENT_SIZE,
            origin=(-1.25, -1/2, 0.125)
        ),
        shapes.cube(
            length=1.125, h=ELEMENT_SIZE,
            origin=(0.125, -1/2, 0.125)
        ), 
    ]
    outer_wall = shapes.cube(
        length=3,
        h=ELEMENT_SIZE, 
        origin=(-1.5, -1.5, -1.5)
    )
    return bempp_cavity.create_grid(outer_wall, *inner_walls) 


def make_grid_9():
    """
    Returns a CavityGrid object of a solid with nine cavities within
    arranged in a 3x3 grid.
    """
    cavity_length = 0.65
    inner_walls = [
        # 1
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(-1.25, -1/2, -1.25)
        ),
        # 2
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(-cavity_length/2, -1/2, -1.25)
        ),
        # 3
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(0.6, -1/2, -1.25)
        ),
        # 4
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(-1.25, -1/2, -cavity_length/2)
        ),
        # 5
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(-cavity_length/2, -1/2, -cavity_length/2)
        ),
        # 6
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(0.6, -1/2, -cavity_length/2)
        ),
        # 7
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(-1.25, -1/2, 0.575)
        ),
        # 8
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(-cavity_length/2, -1/2, 0.575)
        ),
        # 9
        shapes.cube(
            length=cavity_length, h=ELEMENT_SIZE,
            origin=(0.6, -1/2, 0.575)
        ), 
    ]
    outer_wall = shapes.cube(
        length=3,
        h=ELEMENT_SIZE, 
        origin=(-1.5, -1.5, -1.5)
    )
    return bempp_cavity.create_grid(outer_wall, *inner_walls) 

