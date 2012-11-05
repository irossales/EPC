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
    
def singleton_area(t,pd,x):
    value = t(x)
    return map(lambda k:min(value,k), pd) 

number = 1

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


########################### Exercises ############################


#Inputs
#Tensao fundamental(Vf)
def vf_muito_baixa(x):
    return trapezoidFunc(0.00,0.00,0.09,0.12,x)

def vf_baixa(x):
    return trapezoidFunc(0.09,0.12,0.94,0.96,x)

def vf_media(x):
    return trapezoidFunc(0.94,0.96,1.04,1.06,x)

def vf_alta(x):
    return trapezoidFunc(1.04,1.10,1.80,1.80,x)   

#Distorcao harmonica em % (DHT)
def dht_pequena(x):
    return trapezoidFunc(0,0,5,6,x)

def dht_grande(x):
    return trapezoidFunc(5,7,100,100,x)

#Output
#Classe de forma de onda
def classe_interrupcao(x):
    return triangleFunc(0.000,0.167,0.333,x)

def classe_afundamento(x):
    return triangleFunc(0.167,0.333,0.500,x)

def classe_operacao_normal(x):
    return triangleFunc(0.333,0.500,0.667,x)

def classe_elevacao(x):
    return triangleFunc(0.500,0.667,0.883,x)

def classe_harmonicas(x):
    return triangleFunc(0.667,0.833,1.000,x)
    

if __name__ == "__main__":
    my_data = recfromcsv("dados.csv")
    print my_data['amostra']
    print my_data['x1']
