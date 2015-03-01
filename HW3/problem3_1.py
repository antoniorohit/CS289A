#!/usr/bin/env python
"""
Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also contour_image.py.
"""
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.010

#########################################
# answer to part a of the question
###################################
x = np.arange(-4.0, 6.0, delta)
y = np.arange(-2.0, 4.0, delta)
X, Y = np.meshgrid(x, y)
# sigmax, sigmay, mux, muy, sigmaxy
ZA = mlab.bivariate_normal(X, Y, np.sqrt(2.0), 1.0, 1.0, 1.0, 0.0)
# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
plt.grid()
CS = plt.contour(X, Y, ZA)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Solution to part a')
plt.xlim([-5, 6])
plt.ylim([-5, 6])
plt.xlabel("X1")
plt.ylabel("X2")

#########################################
# answer to part b of the question
###################################
x = np.arange(-8.0, 8.0, delta)
y = np.arange(-8.0, 8.0, delta)
X, Y = np.meshgrid(x, y)
# sigmax, sigmay, mux, muy, sigmaxy
ZB = mlab.bivariate_normal(X, Y, np.sqrt(3.0), np.sqrt(2.0), -1.0, 2.0, 1.0)
# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
plt.grid()
CS = plt.contour(X, Y, ZB)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Solution to part b')
plt.xlim([-8, 6])
plt.ylim([-2, 6])
plt.xlabel("X1")
plt.ylabel("X2")

#########################################
# answer to part c of the question
###################################
x = np.arange(-2.0, 4.0, delta)
y = np.arange(-4.0, 6.0, delta)
X, Y = np.meshgrid(x, y)
# sigmax, sigmay, mux, muy, sigmaxy
ZC1 = mlab.bivariate_normal(X, Y, 1.0, np.sqrt(2.0), 0, 2.0, 1.0)
ZC2 = mlab.bivariate_normal(X, Y, 1.0, np.sqrt(2.0), 2.0, 0.0, 1.0)

ZC = ZC1 - ZC2

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
plt.grid()
CS = plt.contour(X, Y, ZC)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Solution to part c')
plt.xlim([-2, 4])
plt.ylim([-4, 6])
plt.xlabel("X1")
plt.ylabel("X2")


#########################################
# answer to part d of the question
###################################
x = np.arange(-2.0, 4.0, delta)
y = np.arange(-4.0, 6.0, delta)
X, Y = np.meshgrid(x, y)
# sigmax, sigmay, mux, muy, sigmaxy
ZD1 = mlab.bivariate_normal(X, Y, 1.0, np.sqrt(2.0), 0, 2.0, 1.0)
ZD2 = mlab.bivariate_normal(X, Y, np.sqrt(3), np.sqrt(2.0), 2.0, 0.0, 1.0)

ZD = ZD1 - ZD2

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
plt.grid()
CS = plt.contour(X, Y, ZD)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Solution to part d')
plt.xlim([-2, 4])
plt.ylim([-4, 6])
plt.xlabel("X1")
plt.ylabel("X2")

#########################################
# answer to part e of the question
###################################
x = np.arange(-4.0, 4.0, delta)
y = np.arange(-6.0, 4.0, delta)
X, Y = np.meshgrid(x, y)
# sigmax, sigmay, mux, muy, sigmaxy
ZE1 = mlab.bivariate_normal(X, Y, 1.0, np.sqrt(2.0), 1.0, 1.0, 0.0)
ZE2 = mlab.bivariate_normal(X, Y, np.sqrt(2.0), np.sqrt(2.0), -1.0, -1.0, 1.0)

ZE = ZE1 - ZE2

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
plt.grid()
CS = plt.contour(X, Y, ZE)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Solution to part e')
plt.xlim([-4, 4])
plt.ylim([-6, 4])
plt.xlabel("X1")
plt.ylabel("X2")


plt.show()