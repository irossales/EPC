import matplotlib
import matplotlib.pylab
import numpy

N_POINTS = 200

# Exercise 1b
def ua(x):
    if 0<=x and x <= 5:
        return 1
    elif 5<x and x<=15:
        return -0.1*x+1.5
    return 0

# Exercise 1b
def ub(x):
    if 5<=x and x<=15:
        return 0.1*x-0.5
    elif 15<x and x<=25:
        return -0.1*x+2.5
    return 0

# Exercise 1b
def uc(x):
    if 15<=x and x<=25:
        return 0.1*x-1.5
    elif 25<x and x<=35:
        return -0.1*x+3.5
    return 0

# Exercise 1b
def ud(x):
    if 25<=x and x<=35:
        return 0.1*x-2.5
    elif 35<x and x<=45:
        return -0.1*x+4.5
    return 0

# Exercise 1b
def ue(x):
    if 35<=x and x<=45:
        return 0.1*x-3.5
    elif 45<x and x<=50:
        return 1
    return 0

# Exercise 1b
def plot_u():
    xa = numpy.arange(0,50,50.0/N_POINTS)
    ya = map(ua, xa)
    xb = numpy.arange(0,50,50.0/N_POINTS)
    yb = map(ub, xb)
    xc = numpy.arange(0,50,50.0/N_POINTS)
    yc = map(uc, xc)
    xd = numpy.arange(0,50,50.0/N_POINTS)
    yd = map(ud, xd)
    xe = numpy.arange(0,50,50.0/N_POINTS)
    ye = map(ue, xe)
    matplotlib.pyplot.scatter(xa, ya, c = "brown", marker = 'o')
    matplotlib.pyplot.scatter(xb, yb, c = "green", marker = 'o')
    matplotlib.pyplot.scatter(xc, yc, c = "pink", marker =  'o')
    matplotlib.pyplot.scatter(xd, yd, c = "blue", marker =  'o')
    matplotlib.pyplot.scatter(xe, ye, c = "black", marker = 'o')
    matplotlib.pyplot.xlabel('x')
    matplotlib.pyplot.ylabel('u(x)')
    matplotlib.pyplot.show()

# Exercise 1c
def active_u(x):
    if ua(x) > 0.0:
        print 'A',
    if ub(x) > 0.0:
        print 'B',
    if uc(x) > 0.0:
        print 'C',
    if ud(x) > 0.0:
        print 'D',
    if ue(x) > 0.0:
        print 'E'

# Exercise 1d
def u(active, x):
    if active == 'A':
        return ua(x)
    elif active == 'B':
        return ub(x)
    elif active == 'C':
        return uc(x)
    elif active == 'D':
        return ud(x)
    elif active == 'E':
        return ue(x)

# Exercise 1e
def crisp(active, value):
    x = numpy.arange(0,50,50.0/N_POINTS)

    if active == 'A':
        return filter (lambda a: ua(a) >= value, x)
    elif active == 'B':
        return filter (lambda a: ub(a) >= value, x)
    elif active == 'C':
        return filter (lambda a: uc(a) >= value, x)
    elif active == 'D':
        return filter (lambda a: ud(a) >= value, x)
    elif active == 'E':
        return filter (lambda a: ue(a) >= value, x)


if __name__ == "__main__":
    print crisp('C', 0.5)
