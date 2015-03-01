import numpy as np
import pylab as plt
from sympy.solvers import solve
from sympy import Symbol
from scipy.optimize import fsolve

#########################################
# answer to part a of the question
###################################
plt.hold('on')

# section a has an easy diagonal covariance. The equations simplify to:
def solve_positive(x2, constant):
    x1 = 1 + np.sqrt(2*(constant-(x2-1)**2))
    return x1

def solve_negative(x2, constant):
    x1 = 1 - np.sqrt(2*(constant-(x2-1)**2))
    return x1

for constant in np.linspace(0.1, 1000, 20):
    # ellipse1 holds one half of the ellipse, ellipse2 the other 
    ellipse1 = []
    ellipse2 = []
    x_a = []
    for x2 in np.linspace(-100, 100, 5000):
        if (2*(constant-(x2-1)**2)) >= 0:   
            sol1, sol2 = solve_negative(x2, constant), solve_positive(x2, constant)
            ellipse1.append(sol1)
            ellipse2.append(sol2)
            x_a.append(x2)
    ellipse_full = ellipse1 + ellipse2
    ellipse_full.append(ellipse1[0])     # without this, the ellipse is not plotted whole
    x_a += x_a[::-1]                    # second half of ellipse - drawn in reverse order
    x_a.append(x_a[0])                  # to match above
    plt.plot(x_a, ellipse_full)

plt.ylim((-50, 50))
plt.xlim((-50, 50))

plt.xlabel("X1")
plt.ylabel("Y1")
plt.title("Part a - Diagonal covariance matrix")

#########################################
# answer to part b of the question
###################################
plt.figure()

mu_a = np.mat([[1], [1]])
sigma_a = np.mat([[2, 0], [0,1]])


def part_b(x2, constant):
    x = Symbol('x')
    return(solve(2*x**2+8*x+3*x2**2-14*x2-2*x*x2-constant, x))

for constant in np.linspace(10, 100, 5):
    ellipse1 = []
    ellipse2 = []
    x_b = []
    for x2 in np.linspace(-10, 10, 500):
        sol1, sol2 = part_b(x2, constant)
        try:
            sol1 = float(sol1)
            sol2 = float(sol2)
            ellipse2.append(sol1)
            ellipse1.append(sol2)            
            x_b.append(x2)
#             print sol1, sol2, constant
        except:
            pass
        
    if ellipse1 != []:
        ellipse1.append(ellipse2[-1])
        ellipse_full =  (ellipse2)+ellipse1[::-1]
        ellipse_full.append(ellipse2[0])
        x_b += x_b[::-1]
        x_b.append(x_b[-1])        
        x_b.append(x_b[-0])        
        plt.plot(x_b, (ellipse_full))

plt.title("Part b - non-diagonal covariance matrix")

#########################################
# answer to part c of the question
###################################
plt.figure()

plt.show()
