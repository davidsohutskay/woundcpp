## mesh generator for a simple quadrilateral domain

import numpy as np

## number of points in the x and y directions
n_x_points = 101
n_y_points = 101

## dimensions of the mesh in the x and y direction
x_init = 0.0
x_final = 50.0
y_init = 0.0
y_final = 50.0

NODES = np.zeros((n_x_points*n_y_points,2))

## create the nodes row by row
for i in range(n_x_points):
	for j in range(n_y_points):
		x_coord = x_init+ i*(x_final-x_init)/(n_x_points-1)
		y_coord = y_init+ j*(y_final-y_init)/(n_y_points-1)
		NODES[j*n_x_points+i,0] = x_coord
		NODES[j*n_x_points+i,1] = y_coord

## create the connectivity
ELEMENTS = np.zeros(((n_x_points-1)*(n_y_points-1),4))
for i in range(n_x_points-1):
	for j in range(n_y_points-1):
		ELEMENTS[j*(n_x_points-1)+i,0] = j*n_x_points+i
		ELEMENTS[j*(n_x_points-1)+i,1] = j*n_x_points+i+1
		ELEMENTS[j*(n_x_points-1)+i,2] = (j+1)*n_x_points+i+1
		ELEMENTS[j*(n_x_points-1)+i,3] = (j+1)*n_x_points+i
		
		