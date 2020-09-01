"""
todo
"""
from bempp.api import shapes

import bempp_cavity

ELEMENT_SIZE = 0.2

def make_grid_1():
    """
    todo
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
    todo
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
    todo
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

