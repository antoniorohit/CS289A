import numpy as np
import pylab as plt
from sympy.solvers import solve
from sympy import Symbol

###############
# a
###############

mu_a = np.mat([[1], [1]])
sigma_a = np.mat([[2, 0], [0,1]])

# p = 0.1

# constant = -2*np.log(2*np.pi*np.sqrt(np.linalg.det(sigma_a))*p)

# print mu_a.shape, sigma_a.shape, constant


def isocontourer(x2, constant):
    x = Symbol('x')
    return(solve(2*x**2+8*x+3*x2**2-14*x2-2*x*x2-constant))
# 
# for constant in [0, 1, 10]:
#     plt.plot(*class2(np.linspace(0, 1, 100), np.linspace(0, 1, 100), constant))
# print "PLOTTED!!"  



def class2(x2, constant):
    x1 = 1 + np.sqrt(2*(constant-(x2-1)**2))
    return x1

def class1(x2, constant):
    x1 = 1 - np.sqrt(2*(constant-(x2-1)**2))
    return x1

plt.hold('on')

for constant in np.linspace(0.1, 1000, 20):
    ellipse1 = []
    ellipse2 = []
    x_a = []
    for x2 in np.linspace(-100, 100, 5000):
        if (2*(constant-(x2-1)**2)) >= 0:   
            sol1, sol2 = class1(x2, constant), class2(x2, constant)
            ellipse1.append(sol1)
            ellipse2.append(sol2)
            x_a.append(x2)
    ellipse1 += (ellipse2)
    ellipse1.append(ellipse1[0])
    x_a += x_a[::-1]
    x_a.append(x_a[0])
    plt.plot(x_a, ellipse1)

plt.ylim((-50, 50))
plt.xlim((-50, 50))


plt.figure()

for constant in np.linspace(0, 200, 10):
    ellipse1 = []
    ellipse2 = []
    x_b = []
    for x2 in np.linspace(-100, 100, 100):
        sol1, sol2 = isocontourer(x2, constant)
        print type(complex(1))
        if (isinstance(complex(sol1), complex) == False) and (isinstance(complex(sol2), complex) == False): 
            print sol1, sol2
            ellipse1.append(sol1)
            ellipse2.append(sol2)            
            x_b.append(x2)
#             print (np.array(ellipse1).flatten()).shape, np.array(x_b).shape
#                 print isocontourer(x2, constant)
        else:
            print sol1, sol2
            
    if ellipse1 != []:
        print ellipse1
        ellipse1 += (ellipse2)
        ellipse1.append(ellipse1[0])
        x_b += x_b[::-1]
        x_b.append(x_b[0])
        print (np.array(ellipse1).flatten()).shape, np.array(x_b).shape
        plt.plot(x_b, (ellipse1))




plt.show()