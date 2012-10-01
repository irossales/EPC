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




########################### Exercises ############################


#input
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
    

#output
def pa(x):
    return trapezoidFunc(0,0,1,3,x)

def pb(x):
    return triangleFunc(1,3,5,x)

def pc(x):
    return triangleFunc(3,5,7,x)

def pd(x):
    return triangleFunc(5,7,9,x)

def pe(x):
    return trapezoidFunc(7,9,10,10,x)



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



# Ex 1
def singleton_area(t,pd,x):
    value = t(x)
    return map(lambda k:min(value,k), pd)
#~ End of Ex 1 
     
def mamdani(ua, ub):
    la = len(ua)
    lb = len(ub)
    return numpy.fromfunction(lambda i,j: numpy.minimum(ua[i],ub[j]),(la,lb),dtype=int)

def zadeh(ua, ub):
    la = len(ua)
    lb = len(ub)
    return numpy.fromfunction(lambda i,j: numpy.maximum(1-ua[i],numpy.minimum(ua[i],ub[j])),(la,lb),dtype=int)

def larsen(ua, ub):
    la = len(ua)
    lb = len(ub)
    return numpy.fromfunction(lambda i,j: ua[i]*ub[j],(la,lb),dtype=int)



def min_max(r, s):
    (r_height,r_width) = r.shape
    (un2,s_width) = s.shape
    print r.shape, s.shape
    result = numpy.empty([r_height,s_width])
    for i in range(r_height):
        for j in range(s_width):
            min_vector = numpy.empty([1,r_width])
            for a in range(r_width):
                min_vector[0][a] = min(r[i][a],s[a][j])
            result[i][j] = numpy.max(min_vector)
    return result
            

number = 1

def plot_graph(x,y):
    global number

    
    matplotlib.pyplot.scatter(x, y, c = "black", marker = 'x')
    
    matplotlib.pyplot.xlabel('x')
    matplotlib.pyplot.ylabel('t(x)')
    matplotlib.pyplot.savefig("graph"+str(number)+".png")
    matplotlib.pyplot.cla()
    number+=1

if __name__ == "__main__":
    xt = discretize(0,50.0,1000)
    xp = discretize(0,10.0,1000)
    
    tad = numpy.array(map(ta, xt))
    tbd = numpy.array(map(tb, xt))
    tcd = numpy.array(map(tc, xt))
    tdd = numpy.array(map(td, xt))
    ted = numpy.array(map(te, xt))
    
    pad = numpy.array(map(pa, xp))
    pbd = numpy.array(map(pb, xp))
    pcd = numpy.array(map(pc, xp))
    pdd = numpy.array(map(pd, xp))
    ped = numpy.array(map(pe, xp))
    
    association={ta:sentence1, tb:sentence2, tc:sentence3, td:sentence4,
                te:sentence5}
                
    discrete_association={ta:tad, tb:tbd, tc:tcd, td:tdd,
                te:ted}
                
    system_association={ta:pad, tb:pbd, tc:pcd, td:pdd,
                te:ped}
                
    def active_sets_and_sentences(t):
        functions = active_list(t,[ta,tb,tc,td,te])        
        print "Active sets:", [f.__name__ for f in functions]
        print "Active sentences:", [association[f].__name__ for f in functions]
    
    # Ex 2 
    for values in [13.3, 18.8, 30.0, 42.3, 47.0]:
        active_sets_and_sentences(values);
    # End of Ex 2
    
    
    def apply_fuzzy_sentence_mamdani(td,pd):
        return min_max(td.reshape(1,len(td)), mamdani(td,pd))    
        
    
    #~ Ex 3
    for values in [13.3, 18.8, 30.0, 42.3, 47.0]:
        for t in active_list(values,[ta,tb,tc,td,te]):
            plot_graph(xt,apply_fuzzy_sentence_mamdani(discrete_association[t],system_association[t]))
    #~ End of Ex 3


    def apply_fuzzy_sentence_zadeh(td,pd):
        return min_max(td.reshape(1,len(td)), zadeh(td,pd))    
            
    
    #~ Ex 4
    for values in [13.3, 18.8, 30.0, 42.3, 47.0]:
        for t in active_list(values,[ta,tb,tc,td,te]):
            plot_graph(xt,apply_fuzzy_sentence_zadeh(discrete_association[t],system_association[t]))
    #~ End of Ex 4
    
    
    def apply_fuzzy_sentence_larsen(td,pd):
        return min_max(td.reshape(1,len(td)), larsen(td,pd))    
    
    
    #~ Ex 5
    for values in [13.3, 18.8, 30.0, 42.3, 47.0]:
        for t in active_list(values,[ta,tb,tc,td,te]):
            plot_graph(xt,apply_fuzzy_sentence_larsen(discrete_association[t],system_association[t]))
    #~ End of Ex 5




