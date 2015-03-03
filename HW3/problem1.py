from numpy.random import choice
import numpy as np
import pylab as plt

NUM_SAMPLES = 100

# the normal function takes mu, sigma (not sigma squared)
# returns NUM_SAMPLES number of normally distributed samples
X1 = np.random.normal(3, 3, NUM_SAMPLES)
X2 = np.random.normal(4, 2, NUM_SAMPLES) + 0.5*X1

# mean of sampled data, rounded to 2 decimals
mu = [np.around(np.mean(X1),2), np.around(np.mean(X2),2)]

###################################
# answer to part a of the question
###################################
print "a) Means of the Sampled Data"
print "   X1:", mu[0], "  X2", mu[1]

# XY matrix - x1 in top row, x2 in bottom row - so that we 
# can calculate covariance using np.cov
X1X2 = np.vstack((X1, X2))

# np.cov takes: A 1-D or 2-D array containing multiple variables and
# observations. Each row of `m` represents a variable, and each column
# a single observation of all those variables.
Covariance = np.around(np.cov(X1X2), 2)

###################################
# answer to part b of the question
###################################
print "b) Covariance of the Sampled Data"
print "  ", Covariance

# calculate eig values and eig vectors
eigenvalues, eigenvectors = np.linalg.eig(Covariance)

###################################
# answer to part c of the question
###################################
print "c) Eigenvalues and Eigenvectors of the Sampled Data"
print "   Eigenvalues:", np.around(eigenvalues, 2) 
print "   Eigenvectors:", np.around(eigenvectors, 2) 

###################################
# answer to part d of the question
###################################
print "d): Data Point Plot for Eigenvectors"
plt.scatter(X1, X2)
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.grid()
plt.hold('on')
plt.arrow(np.mean(X1), np.mean(X2), eigenvalues[0]*eigenvectors[0][0], eigenvalues[0]*eigenvectors[1][0], head_width=0.3, head_length=0.5)
plt.arrow(np.mean(X1), np.mean(X2), eigenvalues[1]*eigenvectors[0][1], eigenvalues[1]*eigenvectors[1][1], head_width=0.3, head_length=0.5)
plt.title("Part d: Data Points and Arrows for Eigenvectors")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("./Results/Q1d_DataPointPlot.png")

plt.figure()

###################################
# answer to part e of the question
###################################
print "e): Rotated Data Point Plot for Eigenvectors"
U = np.mat(eigenvectors)

mu_100 = np.mat(np.ones((2,NUM_SAMPLES)))
mu_100[0:,]*= mu[0]
mu_100[1:,]*= mu[1]

[X1_new, X2_new] = np.transpose(U)*(np.mat(X1X2)-mu_100)
Covariance_new = np.cov(np.vstack((X1_new, X2_new)))
eigvalue_new, eigvect_new = np.linalg.eig(Covariance_new)

plt.scatter(X1_new, X2_new)
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.grid()
plt.hold('on')

plt.arrow(np.mean(X1_new), np.mean(X2_new), eigvalue_new[0]*eigvect_new[0][0], eigvalue_new[0]*eigvect_new[1][0], head_width=0.3, head_length=0.5)
plt.arrow(np.mean(X1_new), np.mean(X2_new), eigvalue_new[1]*eigvect_new[0][1], eigvalue_new[1]*eigvect_new[1][1], head_width=0.3, head_length=0.5)
plt.title("Part e: Rotated Data Points and Arrows for Eigenvectors")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("./Results/Q1e_RotatedDataPointPlot.png")

plt.show()