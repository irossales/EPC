import matplotlib
import matplotlib.pylab
import numpy
    
    
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

def discretize(start, end, number_of_points):
    return numpy.arange(start,end,float((end-start))/number_of_points)
    
def active(u):
    return u > 0.0
    
def active_list(x,u_functions):
    return filter(lambda u: active(u(x)), u_functions)
    
def alpha_cut(a, u):
    return u >= a
    
def algebric_sum(a,b):
    return a+b-a*b

def algebric_product(a,b):
    return a*b    


def max_union(x_values,u_functions):
    union = [0] * len(x_values)
    for u in u_functions:
        union = map(max,union,map(u,x_values))
    return union
    
def min_intersec(x_values,u_functions):
    intersec = [1] * len(x_values)
    for u in u_functions:
        intersec = map(min,intersec,map(u,x_values))
    return intersec

def algebric_sum_union(x_values,u_functions):
    union = [0] * len(x_values)
    for u in u_functions:
        union = map(algebric_sum,union,map(u,x_values))
    return union
    
def algebric_product_intersec(x_values,u_functions):
    intersec = [1] * len(x_values)
    for u in u_functions:
        intersec = map(algebric_product,intersec,map(u,x_values))
    return intersec



def ta(x):
    return trapezoidFunc(0,0,5,15,x)

def tb(x):
    return triangleFunc(5,15,25,x)

def tc(x):
    return triangleFunc(15,25,35,x)

def td(x):
    return triangleFunc(25,35,45,x)

def te(x):
    return trapezoidFunc(35,45,55,55,x)
    
    
# Ex 1
def sentence1(x):
    return pc(x) if active(ta(x)) else 0
    
def sentence2(x):
    return pa(x) if active(tb(x)) else 0
    
def sentence3(x):
    return pd(x) if active(tc(x)) else 0
    
def sentence4(x):
    return pe(x) if active(td(x)) else 0
    
def sentence5(x):
    return pd(x) if active(te(x)) else 0


def plot_u():
    x = discretize(0,50,500)
    ya = map(ta, x)
    yb = map(tb, x)
    yc = map(tc, x)
    yd = map(td, x)
    ye = map(te, x)
    
    
    
    association={ta:sentence1, tb:sentence2, tc:sentence3, td:sentence4,
                te:sentence5}
                
    functions = active_list(40.0,[ta,tb,tc,td,te])        
    
    print "Active sets:", [f.__name__ for f in functions]
    print "Active sentences:", [association[f].__name__ for f in functions]
    
    
    
    
    #~ map(lambda x: x.__name__ , functions) 
    #~ List Comprehensions are easier to understand for someone who
    #~ doesnt know how lambda works
    
    #~ print map(lambda u: alpha_cut(0.5,u), ya)
    
    #~ join = (map(max,ya,yb,yb,yc,yd,ye))
    

    #~ Ex 3a
    #~ functions = []
    #~ for u in [ua,ub,uc,ud,ue]:
        #~ if active(u(18.5)):
            #~ functions.append(u)
    #~ union = max_union(x,functions)
    #~ matplotlib.pyplot.scatter(x, union, c = "black", marker = 'o')

    #~ Ex 3b
    #~ functions = []
    #~ for u in [ua,ub,uc,ud,ue]:
        #~ if active(u(37.29)):
            #~ functions.append(u)
    #~ union = max_union(x,functions)
    #~ matplotlib.pyplot.scatter(x, union, c = "black", marker = 'o')

    #~ Ex 3c
    #~ functions = []
    #~ for u in [ua,ub,uc,ud,ue]:
        #~ if active(u(20.0)):
            #~ functions.append(u)
    #~ union = min_intersec(x,functions)
    #~ matplotlib.pyplot.scatter(x, union, c = "black", marker = 'o')

    #~ Ex 3d
    #~ functions = []
    #~ for u in [ua,ub,uc,ud,ue]:
        #~ if active(u(40.0)):
            #~ functions.append(u)
    
    #~ 3d <- BETTER!
    #~ union = min_intersec(x,functions)
    #~ 
    #~ 4d
    #~ functions = filter(lambda u: active(u(40.0)), [ua,ub,uc,ud,ue])        
    #~ union = algebric_product_intersec(x,functions)
    #~ 
        
    
    
    #~ matplotlib.pyplot.scatter(x, union, c = "black", marker = 'o')
    #~ matplotlib.pyplot.scatter(x, ya, c = "brown", marker = 'o')
    #~ matplotlib.pyplot.scatter(x, yb, c = "green", marker = 'o')
    #~ matplotlib.pyplot.scatter(x, yc, c = "pink", marker =  'o')
    #~ matplotlib.pyplot.scatter(x, yd, c = "blue", marker =  'o')
    #~ matplotlib.pyplot.scatter(x, ye, c = "black", marker = 'o')
    
    #~ matplotlib.pyplot.xlabel('x')
    #~ matplotlib.pyplot.ylabel('u(x)')
    #~ matplotlib.pyplot.show()
    
if __name__ == "__main__":
    plot_u()
