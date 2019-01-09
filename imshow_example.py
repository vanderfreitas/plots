import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure

from matplotlib import rc


# Useful for scientific publications
# Latex font --------------------
rc('text', usetex=True)
font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 18}

rc('font', **font)
params = {'legend.fontsize': 20}
plt.rcParams.update(params)
# -------------------------------




def numpy_array_to_imshow_matrix(x,y,z, x_min, x_max, y_min, y_max, num_bins):
    '''
    Converts 3D coordinates into a matrix with a given dimension
    Returns a 2D matrix with the assigned intensities
    Motivation: When using scatter plots, we end up with a file containing multiple symbols, 
    and the rendering can be costly if the number of points is too high.

    Parameters:
    x - Array with the x coordinates
    y - Array with the y coordinates
    z - Array with the z coordinates
    x_min - Minimum x value to be represented in the 2D matrix
    x_max - Maximum x value to be represented in the 2D matrix
    y_min - Minimum y value to be represented in the 2D matrix
    y_max - Maximum y value to be represented in the 2D matrix
    num_bins - Desired discretization level
    '''

    # Check whether the 
    if(len(x) != len(y) != len(z)):
        raise Exception('x, y and z must have the same dimensions')

    x_bins = np.linspace(x_min, x_max, num_bins)
    y_bins = np.linspace(y_min, y_max, num_bins)

    matrix = np.zeros((num_bins,num_bins))

    # Discretize data according to given bins
    for i in xrange(len(x)):
        index_bin_x = np.digitize(x[i], x_bins)
        index_bin_y = np.digitize(y[i], y_bins)

        matrix[len(y)-index_bin_y-1][index_bin_x-1] = z[i]

    return matrix






# Creating figure
fig = plt.figure( figsize=(3,3) )
ax = fig.add_subplot(111)




# Data
# One could load a data file using:
# data = np.genfromtxt('file.csv', delimiter='\t')
# The data can be organized as:
# x1 y1 z1
# x2 y2 z2
# ...
# xN yN zN
# Then, x = data[:,0], y = data[:,1] and z = data[:,2].


# Desired number of bins
num_points = 1000

# Here we build a simple example: y = x^2, with x in [0,2]
x = np.linspace(0,2,num_points)
y = np.zeros(num_points)
z = np.zeros(num_points) + 1

# Example: Parabola y = x^2
y = x*x

# Convert data from numpy 1D arrays to a 2D matrix, relating x (rows), y (columns) and z (intensity) 
matrix = numpy_array_to_imshow_matrix(x,y,z, 0, 2, 0, 4, num_points)

# Plotting the matrix
ax.imshow(matrix[:,:], interpolation='none', cmap='Greys', aspect='auto', extent = [0, 2, 0, 4])
ax.set_title('Imshow example', fontsize=18)

# One could also, simply use a scatter plot
# For a reasonable number of points, it works perfectly. However, if one wants to plot 
# Thousands of points, the rendering may be costly.
#ax.scatter(x,y, c=z)
#ax.set_title('Scatter example', fontsize=18)

plt.locator_params(axis='x', nbins=3)
plt.locator_params(axis='y', nbins=3)

eixox = r'$x$'
eixoy = r'$y$'


plt.tight_layout()


# Saving the image. One could save as pdf, png, jpg, eps...
# PDF and EPS are preferable for Latex documents 
name = 'imshow_example.pdf' 
fig.savefig(name, dpi=300)



