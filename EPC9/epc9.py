#epc9

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


def active_list_d(x,vectors):
    return filter(lambda u: get_approx_value(u, x, 1.0)>0.0, vectors)
    
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
    z = sum(map(lambda a, b: a*b, y,u))
    return (z/sum(u))

def get_approx_value(vector, index, max_index):
    approx_index = int(len(vector)*index/max_index)
    return vector[approx_index]

########################### Exercises ############################
#Classe de forma de onda
def x1A(x):
    return trapezoidFunc(0, 0, 0.2, 0.6, x)

def x1B(x):
    return triangleFunc(0.1, 0.5, 0.8, x)

def x1C(x):
    return trapezoidFunc(0.4, 0.9, 1.0, 1.0, x)

def x2A(x):
    return trapezoidFunc(0, 0, 0.1, 0.7, x)

def x2B(x):
    return trapezoidFunc(0.3, 0.9, 1.0, 1.0, x)

def x3A(x):
    return triangleFunc(0, 0, 0.5, x)

def x3B(x):
    return trapezoidFunc(0.2, 0.7, 1.0, 1.0, x)

def ycoef(x1,x2,x3,coef):
   return x1*coef[1]+x2*coef[2]+x3*coef[3]+coef[0]


if __name__ == "__main__":
    DISCRETE_POINTS=400
    
    my_data = recfromcsv("dados.csv")
    x1=numpy.vstack(my_data['x1'])
    x2=numpy.vstack(my_data['x2'])
    x3=numpy.vstack(my_data['x3'])
    constant=numpy.vstack(numpy.ones_like(x1))
    
    #~ A = [c , x1, x2, x3]
    f2=my_data['classe']
    f = [0]*len(f2)

    for i in range(0,len(f2)):
        if f2[i]=='A':
            f[i]=0.125
        elif f2[i]=='B':
            f[i]=0.375
        elif f2[i]=='C':
            f[i]=0.625
        elif f2[i]=='D':
            f[i]=0.875

    A = numpy.hstack([constant,x1,x2,x3,numpy.vstack(f)])
    #~  Least-squares solution for A*coeficients=f

    a1 = numpy.array(filter(lambda u: u[1]<=0.6 and u[2]<=0.7 and u[3]<=0.5, A))
    a2 = numpy.array(filter(lambda u: u[1]<=0.6 and u[2]<=0.7 and u[3]>=0.2, A))
    a3 = numpy.array(filter(lambda u: u[1]<=0.6 and u[2]>=0.3 and u[3]<=0.5, A))
    a4 = numpy.array(filter(lambda u: u[1]<=0.6 and u[2]>=0.3 and u[3]>=0.2, A))

    a5 = numpy.array(filter(lambda u: u[1]>=0.1 and u[1]<=0.8 and u[2]<=0.7 and u[3]<=0.5, A))
    a6 = numpy.array(filter(lambda u: u[1]>=0.1 and u[1]<=0.8 and u[2]<=0.7 and u[3]>=0.2, A))
    a7 = numpy.array(filter(lambda u: u[1]>=0.1 and u[1]<=0.8 and u[2]>=0.3 and u[3]<=0.5, A))
    a8 = numpy.array(filter(lambda u: u[1]>=0.1 and u[1]<=0.8 and u[2]>=0.3 and u[3]>=0.2, A))

    a9 = numpy.array(filter(lambda u: u[1]>=0.4 and u[2]<=0.7 and u[3]<=0.5, A))
    a10 = numpy.array(filter(lambda u: u[1]>=0.4 and u[2]<=0.7 and u[3]>=0.2, A))
    a11 = numpy.array(filter(lambda u: u[1]>=0.4 and u[2]>=0.3 and u[3]<=0.5, A))
    a12 = numpy.array(filter(lambda u: u[1]>=0.4 and u[2]>=0.3 and u[3]>=0.2, A))

    #
    coefficients1 = numpy.linalg.lstsq(a1[0:, 0:4], a1[0:, 4])[0]
    coefficients2 = numpy.linalg.lstsq(a2[0:, 0:4], a2[0:, 4])[0]
    coefficients3 = numpy.linalg.lstsq(a3[0:, 0:4], a3[0:, 4])[0]
    coefficients4 = numpy.linalg.lstsq(a4[0:, 0:4], a4[0:, 4])[0]
    coefficients5 = numpy.linalg.lstsq(a5[0:, 0:4], a5[0:, 4])[0]
    coefficients6 = numpy.linalg.lstsq(a6[0:, 0:4], a6[0:, 4])[0]
    coefficients7 = numpy.linalg.lstsq(a7[0:, 0:4], a7[0:, 4])[0]
    coefficients8 = numpy.linalg.lstsq(a8[0:, 0:4], a8[0:, 4])[0]
    coefficients9 = numpy.linalg.lstsq(a8[0:, 0:4], a8[0:, 4])[0]
    coefficients10 = numpy.linalg.lstsq(a8[0:, 0:4], a8[0:, 4])[0]
    coefficients11 = numpy.linalg.lstsq(a8[0:, 0:4], a8[0:, 4])[0]
    coefficients12 = numpy.linalg.lstsq(a8[0:, 0:4], a8[0:, 4])[0]
    #
    print "Regra 1: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients1,"\\\\"
    print "Regra 2: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients2,"\\\\"
    print "Regra 3: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients3,"\\\\"
    print "Regra 4: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients4,"\\\\"
    print "Regra 5: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients5,"\\\\"
    print "Regra 6: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients6,"\\\\"
    print "Regra 7: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients7,"\\\\"
    print "Regra 8: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients8 ,"\\\\"
    print "Regra 9: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients8 ,"\\\\"
    print "Regra 10: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients8 ,"\\\\"
    print "Regra 11: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients8 ,"\\\\"
    print "Regra 12: Valores(d,a,b,c) para f = a*x1 + b*x2 + c*x3 + d):",coefficients8 ,"\\\\"

    x_d = numpy.linspace(0, 1.0, num=DISCRETE_POINTS)

    x1A_d = numpy.array(map(x1A, x_d))
    x1B_d = numpy.array(map(x1B, x_d))
    x1C_d = numpy.array(map(x1C, x_d))
    x2A_d = numpy.array(map(x2A, x_d))
    x2B_d = numpy.array(map(x2B, x_d))
    x3A_d = numpy.array(map(x3A, x_d))
    x3B_d = numpy.array(map(x3B, x_d))
    #
    ##print 'x1A_d', x1A_d
    ##print 'approx', get_approx_value(x1A_d, 0.699999, 1.0)

    exercises = dict([(1,(0.3395, 0.0022, 0.0087)), 
                (2,(0.5102, 0.7464, 0.0860)),
                (3,(0.7382, 0.2647, 0.1916)), 
                (4,(0.0029, 0.3264, 0.2476)), 
                (5,(0.1399, 0.1610, 0.2477)), 
                (6,(0.9430, 0.4476, 0.2648)), 
                (7,(0.8401, 0.4490, 0.2719)), 
                (8,(0.7088, 0.9342, 0.2763)), 
                (9,(0.1957, 0.8423, 0.3085)), 
                (10,(0.0004, 0.6916, 0.5006)), 
                (11,(0.6492, 0.0007, 0.6422)), 
                (12,(0.1283, 0.1882, 0.7253)), 
                (13,(0.2299, 0.1524, 0.7353)), 
                (14,(0.0611, 0.2860, 0.7464)), 
                (15,(0.1903, 0.6523, 0.7820)), 
                (16,(0.2225, 0.9182, 0.7820)), 
                (17,(0.6423, 0.3229, 0.8567)), 
                (18,(0.3879, 0.1307, 0.8656)), 
                (19,(0.8882, 0.3077, 0.8931)), 
                (20,(0.1818, 0.5078, 0.9046))]) 

    for h in range(1,21):
        for (x1ex, x2ex, x3ex) in [exercises[h]]:

        #    #plot_graph(x_d, active_list_d(x1, [x1A_d, x1B_d])[0])

            active_x1 = active_list_d(x1ex, [x1A_d, x1B_d, x1C_d])
            active_x2 = active_list_d(x2ex, [x2A_d, x2B_d])
            active_x3 = active_list_d(x3ex, [x3A_d, x3B_d])

            u_final = []
            y_final = []

            for ax1 in active_x1:
                for ax2 in active_x2:
                    for ax3 in active_x3:
                        using_conective_d = min(get_approx_value(ax1, x1ex, 1.0), get_approx_value(ax2, x2ex, 1.0), 
                            get_approx_value(ax3, x3ex, 1.0))
                        u_final.append(using_conective_d)

                        if numpy.allclose(ax1,x1A_d) and numpy.allclose(ax2,x2A_d) and numpy.allclose(ax3,x3A_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients1)
                            print 'Regra 1 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)

                        elif numpy.allclose(ax1,x1A_d) and numpy.allclose(ax2,x2A_d) and numpy.allclose(ax3,x3B_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients2)
                            print 'Regra 2 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1A_d) and numpy.allclose(ax2,x2B_d) and numpy.allclose(ax3,x3A_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients3)
                            print 'Regra 3 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1A_d) and numpy.allclose(ax2,x2B_d) and numpy.allclose(ax3,x3B_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients4)
                            print 'Regra 4 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1B_d) and numpy.allclose(ax2,x2A_d) and numpy.allclose(ax3,x3A_d):                        
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients5)
                            print 'Regra 5 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1B_d) and numpy.allclose(ax2,x2A_d) and numpy.allclose(ax3,x3B_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients6)
                            print 'Regra 6 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1B_d) and numpy.allclose(ax2,x2B_d) and numpy.allclose(ax3,x3A_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients7)
                            print 'Regra 7 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1B_d) and numpy.allclose(ax2,x2B_d) and numpy.allclose(ax3,x3B_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients8)
                            print 'Regra 8 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1C_d) and numpy.allclose(ax2,x2A_d) and numpy.allclose(ax3,x3A_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients9)
                            print 'Regra 9 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1C_d) and numpy.allclose(ax2,x2A_d) and numpy.allclose(ax3,x3B_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients10)
                            print 'Regra 10 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1C_d) and numpy.allclose(ax2,x2B_d) and numpy.allclose(ax3,x3A_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients11)
                            print 'Regra 11 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
                        elif numpy.allclose(ax1,x1C_d) and numpy.allclose(ax2,x2B_d) and numpy.allclose(ax3,x3B_d):
                            contribuicao = ycoef(x1ex, x2ex, x3ex, coefficients12)
                            print 'Regra 12 , y da regra:',contribuicao, 'peso da regra', using_conective_d , "\\\\"
                            y_final.append(contribuicao)
                            
            answer = weighted_mean(y_final, u_final)
            print 'Resposta y=', answer
            if 0<=answer and answer<0.25:
                print 'A'
            elif answer<0.50:
                print 'B'
            elif answer<0.75:
                print 'C'
            else:
                print 'D'
            
            print 'Fim exercicio'
        
