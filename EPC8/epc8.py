#epc7

import matplotlib
import matplotlib.pylab
import numpy
import csv
from numpy import recfromcsv
    
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

def mamdani(ua, ub):
    la = len(ua)
    lb = len(ub)
    saida = numpy.fromfunction(lambda i,j: numpy.minimum(ua[i],ub[j]),(la,lb),dtype=int)
    return saida

def zadeh(ua, ub):
    la = len(ua)
    lb = len(ub)
    return numpy.fromfunction(lambda i,j:
        numpy.maximum(1-ua[i],numpy.minimum(ua[i],ub[j])),(la,lb),dtype=int)

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
    
def plot_graph(x,y):
    global number
    
    matplotlib.pyplot.scatter(x, y, c = "black", marker = 'x')
    
    matplotlib.pyplot.xlabel('x')
    matplotlib.pyplot.ylabel('p\'(x)')
    matplotlib.pyplot.axis([min(x),max(x),min(y),max(y)])
    #~ matplotlib.pyplot.savefig("graph"+str(number)+".png")
    matplotlib.pyplot.show();
    matplotlib.pyplot.cla()
    number+=1

def weighted_mean(y, u):
    return map(lambda a, b: a*b, y, u)/sum(u)

def get_approx_value(vector, index, max_index):
    approx_index = int(len(vector)*index/max_index)
    return vector[approx_index]

########################### Exercises ############################
#Classe de forma de onda
def x1A(x):
    return trapezoidFunc(0, 0, 0.3, 0.7, x)

def x1B(x):
    return trapezoidFunc(0.3, 0.7, 1.0, 1.0, x)

def x2A(x):
    return trapezoidFunc(0, 0, 0.25, 0.75, x)

def x2B(x):
    return trapezoidFunc(0.25, 0.75, 1.0, 1.0, x)

def x3A(x):
    return triangleFunc(0, 0, 0.5, x)

def x3B(x):
    return trapezoidFunc(0, 0.8, 1.0, 1.0, x)

if __name__ == "__main__":
    DISCRETE_POINTS=1000
    
    my_data = recfromcsv("dados.csv")
    print my_data['amostra']
    print my_data['x1']

    x_d = numpy.linspace(0, 1.0, num=DISCRETE_POINTS)

    x1A_d = numpy.array(map(x1A, x_d))
    x1B_d = numpy.array(map(x1B, x_d))
    x2A_d = numpy.array(map(x2A, x_d))
    x2B_d = numpy.array(map(x2B, x_d))
    x3A_d = numpy.array(map(x3A, x_d))
    x3B_d = numpy.array(map(x3B, x_d))
    
    print 'x1A_d', x1A_d
    print 'approx', get_approx_value(x1A_d, 0.699999, 1.0)


   # #input discretization
   # x_vf = numpy.linspace(0,1.8,num=DISCRETE_POINTS)
   # x_dht = numpy.linspace(0,100,num=DISCRETE_POINTS)
   # #output discretization
   # x_classe = numpy.linspace(0,1.0,num=DISCRETE_POINTS)
   # 
   # #discretized membership functions 
   # vf_muito_baixa_d = numpy.array(map(vf_muito_baixa, x_vf))
   # vf_baixa_d = numpy.array(map(vf_baixa, x_vf))
   # vf_media_d = numpy.array(map(vf_media, x_vf))
   # vf_alta_d = numpy.array(map(vf_alta, x_vf))
   # 
   # #discretized membership functions 
   # dht_pequena_d = numpy.array(map(dht_pequena, x_dht))
   # dht_grande_d = numpy.array(map(dht_grande, x_dht))
   # 
   # #discretized membership functions 
   # classe_interrupcao_d = numpy.array(map(classe_interrupcao, x_classe))
   # classe_afundamento_d = numpy.array(map(classe_afundamento, x_classe)) 
   # classe_operacao_normal_d = numpy.array(map(classe_operacao_normal, x_classe))
   # classe_elevacao_d = numpy.array(map(classe_elevacao, x_classe))
   # classe_harmonicas_d = numpy.array(map(classe_harmonicas, x_classe))
   # 

