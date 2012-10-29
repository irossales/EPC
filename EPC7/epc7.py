#epc7

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

     
def mamdani(ua, ub):
    la = len(ua)
    lb = len(ub)
    saida = numpy.fromfunction(lambda i,j: numpy.minimum(ua[i],ub[j]),(la,lb),dtype=int)
    return saida

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
    DISCRETE_POINTS=200
    #input discretization
    x_vf = numpy.linspace(0,1.8,num=DISCRETE_POINTS)
    x_dht = numpy.linspace(0,100,num=DISCRETE_POINTS)
    #output discretization
    x_classe = numpy.linspace(0,1.0,num=DISCRETE_POINTS)
    
    #discretized membership functions 
    vf_muito_baixa_d = numpy.array(map(vf_muito_baixa, x_vf))
    vf_baixa_d = numpy.array(map(vf_baixa, x_vf))
    vf_media_d = numpy.array(map(vf_media, x_vf))
    vf_alta_d = numpy.array(map(vf_alta, x_vf))
    
    #discretized membership functions 
    dht_pequena_d = numpy.array(map(dht_pequena, x_dht))
    dht_grande_d = numpy.array(map(dht_grande, x_dht))
    
    #discretized membership functions 
    classe_interrupcao_d = numpy.array(map(classe_interrupcao, x_classe))
    classe_afundamento_d = numpy.array(map(classe_afundamento, x_classe)) 
    classe_operacao_normal_d = numpy.array(map(classe_operacao_normal, x_classe))
    classe_elevacao_d = numpy.array(map(classe_elevacao, x_classe))
    classe_harmonicas_d = numpy.array(map(classe_harmonicas, x_classe))
    
    #Table
    #                  DHT
    #       Pequena            Grande 
    #   MB  interrupcao        interrupcao
    #V  B   afundamento        harmonicos
    #f  M   operacao nornal    harmonicos
    #   A   elevacao           harmonicos
    
    table = dict([((dht_pequena,vf_muito_baixa),classe_interrupcao_d),
             ((dht_pequena,vf_baixa),classe_afundamento_d),
             ((dht_pequena,vf_media),classe_operacao_normal_d),
             ((dht_pequena,vf_alta),classe_elevacao_d),
             ((dht_grande,vf_muito_baixa),classe_interrupcao_d),
             ((dht_grande,vf_baixa),classe_harmonicas_d),
             ((dht_grande,vf_media),classe_harmonicas_d),
             ((dht_grande,vf_alta),classe_harmonicas_d)])
             
    exercices = dict([(1,(0.01,0.34)),
                     (2,(0.05,16.26)),
                     (3,(0.50,4.84)),
                     (4,(0.85,1.79)),
                     (5,(1.02,0.47)),
                     (6,(0.97,1.21)),
                     (7,(1.57,4.76)),
                     (8,(1.26,1.21)),
                     (9,(0.99,16.32)),
                     (10,(1.20,18.96))])
                
                
    (vf_value,dht_value) = exercices[1]
    
    
    active_vfs = active_list(vf_value,[vf_muito_baixa,vf_baixa,vf_media,vf_alta]);
    active_dhts = active_list(dht_value,[dht_pequena,dht_grande])
    
    valor_final = 0

    #All combinations of active functions
    for active_vf in active_vfs:
        for active_dht in active_dhts:
            #Using min operator as conective "e"
            using_conective_e = min(active_vf(vf_value),active_dht(dht_value))
            print "Value to use in mamdani:",using_conective_e
            output_function_d = table[(active_dht,active_vf)]
            after_mamdani = numpy.minimum(using_conective_e,output_function_d)
            #plot_graph(x_classe,after_mamdani)
            #agregacao
            valor_final = numpy.maximum(valor_final, after_mamdani)
    
    #~ print "Final value:", valor_final
    max_values_position = map(lambda u: False if u <> max(valor_final) else True, valor_final)
   
    
    total_sum=0
    
    for i in range(len(max_values_position)):
        if max_values_position[i]:
            total_sum+=x_classe[i]

            
    class_value = total_sum/numpy.sum(max_values_position)

    print class_value
    
            #~ print vf_value, "->", active_vf, "->", active_vf(vf_value)
            #~ print dht_value, "->", active_dht, "->", active_dht(dht_value)
        
    y = 0.5
    if 0.00<y and y<=0.25:
        print "interrupcao"
    elif y<=0.42:
        print "afundamento"
    elif y <= 0.58:
        print "operacao normal"
    elif y<=0.75:
        print "elevacao"
    elif y<=1.00:
        print "harmonicas"
    
    
    
                     

