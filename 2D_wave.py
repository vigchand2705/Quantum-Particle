from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy import *
from pylab import *
from numpy.linalg import solve

# Converts an NxN 2D array to a 1D array of size N**2 
def to1D(arr, N):
	new_arr = zeros(N**2, complex)
	for i in range(N):
		new_arr[i*N:(i+1)*N] = arr[i,:]
	return new_arr

# Converts a 1D array of size N**2 to an NxN 2D array 
def to2D(arr, N):
	new_arr = empty([N, N], complex)
	for i in range(N):
		new_arr[i,:] = arr[i*N:(i+1)*N]

	return new_arr
# Return the value of the intitial wave packet at postion x
def initial_waveft(x, y):
	return A*((2/side)*sin(pi*x/side)*sin(pi*y/side) + (4/side)*sin(2*pi*x/side)*sin(2*pi*y/side))

#Constants
M = 9.109e-31 		# Mass of electron in kg
hbar = 1.0546e-34  	# Planck's constant over 2*pi
L = 1.0e-8			# Length of box in metres
N = 25				# Number of slices
a = L/N**2 			# Length of spatial slice in metres
h = 5.0e-18			# Timestep in seconds
tend = 500*h 		
A = 0.8e-10			# Normalization Constant
side = a*N 			# side length of the box

#Constants for matrices
a1 = 2*(1 + ((1j*h*hbar)/(2*M*a**2)))
a2 = (-1j*h*hbar)/(4*M*a**2)
b1 = 2*(1 - ((1j*h*hbar)/(2*M*a**2)))
b2 = (1j*h*hbar)/(4*M*a**2)

# Create the initial wave packet using the helper function
psi_t2D = zeros([N,N], complex)
for i in range(N):
	for j in range(N):
		if (i != 0) and (j != 0) and (i != N -1) and (j != N - 1):
			psi_t2D[i,j] = initial_waveft(i*a, j*a)

# convert to 1D array to perform Crank-Nicholson
psi_t = to1D(psi_t2D, N)	

# Create matrix A
A = zeros([N**2,N**2], complex)

for i in range(N**2):
	for j in range(N**2):
		if i == j:
			A[i, i] = a1
			if i+1 < N**2:	
				A[i+1, i] = a2
				A[i, i+1] = a2
			if i+N < N**2:
				A[i+N, i] = a2
				A[i, i+N] = a2

t = 0
fig = plt.figure()

while t < tend:
	clf() #clear current figure
		
	# Update v using the matrix B and psi(t)
	v = zeros(N**2, complex)

	for i in range(N**2):
		if (i < N) or (i > N**2 - N - 1) or (i%N == 0) or (i%N == N-1):
			v[i] = psi_t[i]
		else:
			v[i] = b1*psi_t[i] + b2*(psi_t[i+1] + psi_t[i-1] + psi_t[i+N]+ psi_t[i-N])
	
	# solve for psi(t+h)
	psi = solve(A, v)
	psi_t = psi
	
	# convert to a 2D array for plotting
	psi2D = to2D(psi, N)
	
	# set up 3D axes
	ax = fig.gca(projection='3d')
	x = linspace(0, side, N)
	y = linspace(0, side, N)
	X, Y = meshgrid(x, y)
	
	# make surface plot
	surf = ax.plot_surface(X, Y, psi2D.real, rstride=1, cstride=1)
	my_title="Time: %e" % (t)
	title(my_title)
	ax.set_zlim(-1.0, 1.0)
	pause(0.01)
	t += h
show()


