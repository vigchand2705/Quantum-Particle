####################################################################
# This program uses the Crank-Nicolson scheme to solve the 1D
# Schrodinger equation with no potential energy
####################################################################

from banded import banded
from numpy import *
from pylab import *

###################################################################
# user defined functions
###################################################################

#define makematrixA to make matrix A
def makematrixA(a1,a2,x):
    A_subdiag=zeros(N,complex)
    A_subdiag[:N-2] = a2

    A_diag=zeros(N,complex)
    A_diag[1:N-1] = a1+1j*h*V(x[1:N-1])/(2*hbar)
    A_diag[0]=1.0
    A_diag[N-1]=1.0

    A_supdiag=zeros(N,complex)
    A_supdiag[2:] = a2
    
    #tridiagonal array stored in compact form for banded.py function
    A=[A_supdiag, A_diag, A_subdiag]
    return A

#####################################################################

#define rhs matrix multiplication v = B*psi
def rhs(psi,N,b1,b2):
    v = zeros(N,complex)
    # boundary conditisions on rhs = B * psi
    v[0]=psi[0]
    v[N-1]=psi[N-1]
    v[1:N-1]=(b1-1j*h*V(x[1:N-1])/(2*hbar))*psi[1:N-1]+b2*(psi[2:]+psi[:-2])
    return v

#define potential
def V(x):
    N = len(x)
    VOut = zeros(N,float)
    VOut[x>L/2.0]=V_0
    return VOut

######################################################################
# main program starts here
######################################################################

# Define Constants
M = 9.109e-31   # mass of electron [kg]
L = 2.e-8       # length of box [m]
N = 5000        # number of spatial grid points
a = L/float(N)  # Grid spacing [m]
h = 1.e-18      # time step [s]
x0 = 0.45*L       # constant in initial condition [m]
sigma = 1.e-10  # constant in initial condition [m]
kappa = 5.e10   # constant in initial condition [m^-1]
eV = 1.6e-19
#V0= -40*eV #
V_0= 40*eV #
#V0= 0.0#
#coefficients in A and B matrices
hbar = 1.05457173e-34       # Planck's constant in Js
fac = 1j*hbar/(4.*M*a*a)    # constant in A
a1 = 1.0+2.*h*fac           # diagonal entries of A
a2 = -h*fac                 # sub and sup diagonal entries of A
b1 = 1-2.*h*fac             # diagonal entries of B
b2 = h*fac                  # sub and sup diagonal entries of B


#initialize psi and x arrays
psi = zeros(N, complex)
x = linspace(0,L,N)

#initial condition
psi[0]=0.0
psi[N-1]=0.0
psi[1:N-2] = exp(-(x[1:N-2]-x0)**2/(2.0*pow(sigma,2)))*exp(1j*kappa*x[1:N-2])
#psi[1:N-2] = sqrt(2/L)*sin((pi/L)*x[1:N-2])

# needed for animation and plotting
plot_times=array([0,1.e-16,1.e-15]) # times at which to save plot of psi

# turn on interactive mode (for plotting)
ion()

#how often to refresh plot
animation_timestep=h

# Set up plotting figure
fig = figure(len(plot_times)+1)
suptitle('Time Dep. 1D Schrodinger: Lin System Soln')
ax = axes(xlim=(x.min(),x.max()), ylim=((psi.real).min(),(psi.real).max()))
#line, = plot(x, abs(psi),'-b',x,psi.real,'-r',x,-abs(psi),'-b')
my_title="Time: %e" % (0)
ax.set_title(my_title)
ax.set_xlabel('x')
ax.set_ylabel('Re(psi)')

# define subdiagonal, diagonal and superdiagonal of lhs "A" matrix
A = makematrixA(a1,a2,x)

# initialize t
tstart = 0.0    # start time
tend = 1.0e-15   # end time
t = tstart
print_interval = 50
step = 0
print_num = 0
while t < tend:
    clf()
    # multiply matrix equation v = B * psi
    v = rhs(psi,N,b1,b2)

    # solve linear system A psi = v for psi
    psi=banded(A,v,1,1)
    
    # update animation with psi at current timestep    
    plot(x, abs(psi),'-b',x,psi.real,'-r',x,-abs(psi),'-b')
    plot(x,V(x)/eV/80.0,'g')
    ylim((-1,1))
    my_title="Time: %e" % (t)
    title(my_title)
    if step%print_interval==0:
		savefig('Lab09-Q2b-%i.png'%print_num)   
		print_num += 1   
    step+=1
    pause(0.01)
	
        
    t +=h


show()

