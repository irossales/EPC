#epc10

import matplotlib
import matplotlib.pylab
import numpy

number = 0

def plot_graph(x,y):
    global number
    
    matplotlib.pyplot.scatter(x, y, c = "black", marker = 'x')
    
    matplotlib.pyplot.xlabel('x')
    matplotlib.pyplot.ylabel('u(x)')
    matplotlib.pyplot.axis([min(x),max(x),min(y),max(y)])
    matplotlib.pyplot.savefig("graph"+str(number)+".png")
    #matplotlib.pyplot.show();
    matplotlib.pyplot.cla()
    number+=1

def trapezoidFunc(a, m, n, b, x):
    if x < a: 
        return 0
    elif x < m:
        return(x-a)/(m-a)
    elif x < n:
        return 1
    elif x < b:  
        return (b-x)/(b-n)
    else:
        return 0

def triangleFunc(a, m, b, x):
    return trapezoidFunc(a,m,m,b,x)

def A1(x):
    return triangleFunc(1.0, 2.0, 4.0, x)

def A2(x):
    return triangleFunc(1, 2.5, 4, x)

def A3(x):
    return triangleFunc(1, 3.0, 4, x)

def B1(x):
    return triangleFunc(3,4,8,x)

def B2(x):
    return triangleFunc(3,5,8,x)

def B3(x):
    return triangleFunc(3,7,8,x)

def alpha_cut(a, u):
    return u >= a or numpy.allclose(u,a,atol=0.01)

DISCRETE_POINTS=1000
ALPHA_CUTS = 200

def min_alpha(a, e):
    #return min(filter(lambda u: alpha_cut(a, u)==True, e)) 
    return min(numpy.where(numpy.array(map(lambda u: alpha_cut(a, u)==True, e))==True)[0])*8.0/(DISCRETE_POINTS-1)

def max_alpha(a, e):
    return max(numpy.where(numpy.array(map(lambda u: alpha_cut(a, u)==True, e))==True)[0])*8.0/(DISCRETE_POINTS-1)

if __name__ == "__main__":

    x_d = numpy.linspace(0, 8.0, num=DISCRETE_POINTS)

    A1_d = numpy.array(map(A1, x_d))
    A2_d = numpy.array(map(A2, x_d))
    A3_d = numpy.array(map(A3, x_d))
    B1_d = numpy.array(map(B1, x_d))
    B2_d = numpy.array(map(B2, x_d))
    B3_d = numpy.array(map(B3, x_d))

    alpha = numpy.linspace(0,1.0, num = ALPHA_CUTS)
   
    add = [0]*ALPHA_CUTS*2
    subb = [0]*ALPHA_CUTS*2
    mult = [0]*ALPHA_CUTS*2
    div = [0]*ALPHA_CUTS*2
    op_min = [0]*ALPHA_CUTS*2
    op_max = [0]*ALPHA_CUTS*2
    c_d = [0]*ALPHA_CUTS*2

    for A in [A1_d, A2_d, A3_d]:
        for B in [B1_d, B2_d, B3_d]:
    #for A in [A1_d]:
    #    for B in [B1_d]:
            for i in range(0,len(alpha)):
                c_d[i] = alpha[i]
                c_d[ALPHA_CUTS*2-i-1] = alpha[i]
                a1 = min_alpha(alpha[i], A)
                b1 = min_alpha(alpha[i],B)
                a2 = max_alpha(alpha[i],A)
                b2 = max_alpha(alpha[i],B)
                add[i] = a1+b1
                add[ALPHA_CUTS*2-i-1] = a2+b2
                subb[i] = a1-b1
                subb[ALPHA_CUTS*2-i-1] = a2-b2
                mult[i] = a1*b1
                mult[ALPHA_CUTS*2-i-1] = a2*b2
                if b2==0:
                    div[i]=0
                else:
                    div[i] = a1/b2
                if b1==0:
                    div[ALPHA_CUTS*2-i-1] = 0
                else:
                    div[ALPHA_CUTS*2-i-1] = a2/b1
                op_min[i] = min(a1,b1)
                op_min[ALPHA_CUTS*2-i-1] = min(a2,b2)
                op_max[i] = max(a1,b1)
                op_max[ALPHA_CUTS*2-i-1] = max(a2,b2)

            plot_graph(add, c_d)
            plot_graph(subb, c_d)
            plot_graph(mult, c_d)
            plot_graph(div, c_d)
            plot_graph(op_min, c_d)
            plot_graph(op_max, c_d)

