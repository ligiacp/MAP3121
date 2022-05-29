# EP1 - Cálculo Numérico (MAP3121-2022)
# Bruno Matutani Santos - 11804682
# Ligia Corunha Palma - 11352268
#
#
#
#-----------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import sys,os,math

#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------Decomposição LU---------------------------------------------------------

def decLU(A,n):
    U = np.zeros((n,n))
    L = np.eye(n)

    for i in range(n): #É o n
        i_comp = i+1 #Usado para comparacoes
        if (i_comp==1):
            #Iteração 0 (primeira iteração)
            U[i, i:n] = A[i,i:n]
            L[(i+1):n,i] = (1/(U[i,i])) * ( A[(i+1):n,i])

        elif (i_comp!=1):
            #A partir da iteração 1 (segunda iteração)
            U[i, i:n] = A[i,i:n] - np.matmul(L[i, 0:(i)], U[0:(i), i:n])

            if (i_comp!=n): #Se i de comparação (número de iterações) for diferente de n
                L[(i+1):n,i] =  (1/ (U[i,i])) * (A[(i+1):n,i] - (np.matmul(L[(i+1):n, 0:(i)],U[0:(i),i])))
    return (L,U)



#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------Matrizes Tridiagonais----------------------------------------------------

def MatrizesTridiagonais(A, d, n_tam):
    #---Criação dos vetores a,b e c (separação de A em 3 vetores) 
    a = np.array([]) #Arrays Unidimensionais
    b = np.array([])
    c = np.array([])
    a = np.append(a,[0]) #Adicionando o primeiro elemento de a
    for i in range(n_tam):
        for j in range (n_tam):
            if (i==j): #Elemento da diagonal principal
                b = np.append(b,A[i,j]) #Adiciona elemento no fim da lista 
            if (i == j-1):
                c = np.append(c,A[i,j])
            if (i == j+1):
                a = np.append(a,A[i,j])

    c = np.append(c,0)

    #---Decomposição LU da matriz tridiagonal
    u = np.array([b[0]])
    l = np.array([0])

    for v in range(1,n_tam):
        l = np.append(l,a[v]/u[v-1])
        u = np.append(u,b[v] -(l[v]*c[v-1]) )



    #---Solucao do sistema de equações
    y = np.array([d[0]])

    #Cálculo de y
    for k in range(1,n_tam):
        y = np.append(y, d[k]- (l[k]*y[k-1]))


    #Cálculo de x (resultado)
    x = np.zeros(n_tam)
    x[n_tam-1] = y[n_tam-1]/u[n_tam-1]

    for h in reversed(range(0,n_tam-1)):
        x[h]= (y[h]-c[h]*x[h+1])/(u[h])

    return x


#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------Sistemas Tridiagonais Cíclicos------------------------------------------


def SistemasTridiagonaisCiclicos(A, d, n):
    #---Criação da matriz tridiagonal cíclica
    a = np.array([]) #Arrays Unidimensionais
    b = np.array([])
    c = np.array([])
    d = np.array([])


    #Calculando e adicionando valores em a[]
    for i_a in range (n-1): #de 1 a 19
        i_a_comp = i_a + 1
        a = np.append(a, ((2*i_a_comp - 1)/(4*i_a_comp))) #Adiciona elemento no fim da lista 

    a = np.append (a, (2*n - 1) / (2*n)) 


    #Calculando e adicionando valores em b[]
    for i_b in range ((n)): #de 1 a 20
        i_b_comp = i_b + 1
        b = np.append(b, (2)) #Adiciona elemento no fim da lista 


    #Calculando e adicionando valores em c[]
    for i_c in range ((n)): #de 1 a 20
        i_c_comp = i_c + 1
        c = np.append(c, (1 - (a[i_c]))) #Adiciona elemento no fim da lista 


    #Calculando e adicionando valores em d[]
    for i_d in range ((n)): #de 1 a 20
        i_d_comp = i_d + 1
        d = np.append( d, np.cos((2*(np.pi)*np.square(i_d_comp))/(np.square(n))))

    #--- Cálcilo de valores intermediários
    
    #Criação de A[]
    A = np.zeros((n,n))
    A[0,n-1] = a[0]
    A[n-1,0] = c[n-1]
    #Preenchimento de A[]
    for i in range (n):
        for j in range (n):
            if (i == j):
                A[i, j] = b[i]
            if (i == j-1):
                A[i,j] = c[i]
            if (i == j+1):
                A[i,j] = a[i]



    #Criação de T[]
    T = A[0:(n-1), 0:(n-1)]



    #Preenchimento de v[]
    v = np.array([])
    v = np.append(v, a[0])
    for i_v in range (1, n - 2): #de 1 a 19
        v = np.append(v, (0))
    v = np.append (v, c[n-2])



    #Criação e preenchimento de w[]
    w = np.array([])
    w = np.append(w, c[n-1])
    for i_w in range (1, n - 2): #de 1 a 19
        w = np.append(w, (0))
    w = np.append (w, a[n-1])


    #Criação e preenchimento de d_linha[]
    d_linha = np.array([])
    for i_d_linha in range(n-1):
        d_linha = np.append (d_linha , d[i_d_linha])

    #--- Cálculo de x (resultado final)

    x = np.array([])
    y_linha = np.array([])
    z_linha = np.array([])
    y_linha = MatrizesTridiagonais(T, d_linha, n-1)
    z_linha = MatrizesTridiagonais(T, v, n-1)


    x_n = (d[n-1] - (c[n-1]*y_linha[0]) - (a[n-1] * y_linha[n-2] )) / (b[n-1] - (c[n-1]*z_linha[0]) - (a[n-1]*z_linha[n-2] ))

    for var in range (n-1):
        x = np.append(x, y_linha[var] - x_n*z_linha[var])

    x = np.append(x, x_n)

    return x

#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------main--------------------------------------------------------------------

print("Escolha uma das seguintes opções, digitando o numero: ")
print("1) Decomposição LU")
print("2) Solução de matrizes tridiagonais")
print("3) Solução de sistemas tridiagonais cíclicos")
print("Opção: ")

opcao1 = int(input())

if (opcao1 == 1):
    print("Digite a dimensão da matriz quadrada n x n: ")
    n = int(input())
    print("Escreva os elementos da matriz: ")
    A = np.zeros((n,n))
    for i in range (n):
        for j in range (n):
            print("Elemento na posição [" + str(i+1) + "," + str(j+1) + "]")
            A[i,j] = input()
    L, U = decLU(A,n)
    print("\n \n \n")
    print ("L:" )
    print (L)
    print ("U: ")
    print (U)

if (opcao1 == 2):
    print("Digite a dimensão da matriz quadrada n x n: ")
    n = int(input())
    print("Escreva os elementos da matriz: ")
    A = np.zeros((n,n))
    for i in range (n):
        for j in range (n):
            print("Elemento na posição [" + str(i+1) + "," + str(j+1) + "]")
            A[i,j] = input()
    print("Digite o vetor d a ser utilizado: ")
    d = np.zeros(n)
    for i in range (n):
        print("Elemento na posição [" + str(i+1) + "]")
        d[i] = int(input())
    x = MatrizesTridiagonais(A, d, n)
    print ("\n \n \n")
    print ("X: ")
    print (x)

if (opcao1 == 3):
    print("Escolha: ")
    print("1) Usar o caso teste do enunciado")
    print("2) Usar o caso teste, porém com outra dimensão")
    print("3) Escolher os valores e o tamanho da matriz")
    print("\n \n")
    print("Opção: ")
    opcao2 = int(input())

    if (opcao2 == 1):
            #-------------------------------criação das entradas
        a = np.array([]) #Arrays Unidimensionais
        b = np.array([])
        c = np.array([])
        d = np.array([])
        n = 20

        #criando valores para a[]
        for i_a in range (n-1): #de 1 a 19
            i_a_comp = i_a + 1
            a = np.append(a, ((2*i_a_comp - 1)/(4*i_a_comp))) #Adiciona elemento no fim da lista 

        a = np.append (a, (2*n - 1) / (2*n)) 


        #criando valores para b[]
        for i_b in range ((n)): #de 1 a 20
            i_b_comp = i_b + 1
            b = np.append(b, (2)) #Adiciona elemento no fim da lista 


        #criando valores para c[]
        for i_c in range ((n)): #de 1 a 20
            i_c_comp = i_c + 1
            c = np.append(c, (1 - (a[i_c]))) #Adiciona elemento no fim da lista 


        #criando os valores para d[]
        for i_d in range ((n)): #de 1 a 20
            i_d_comp = i_d + 1
            d = np.append( d, np.cos((2*(np.pi)*np.square(i_d_comp))/(np.square(n))))

        #criação de A[]
            
        A = np.zeros((n,n))
        A[0,n-1] = a[0]
        A[n-1,0] = c[19]
        for i in range (n):
            for j in range (n):
                if (i == j):
                    A[i, j] = b[i]
                if (i == j-1):
                    A[i,j] = c[i]
                if (i == j+1):
                    A[i,j] = a[i]
        x = SistemasTridiagonaisCiclicos(A, d, n)
        print()
        print()
        print()
        print ("X: " )
        for i in range (n):
            print("X[" + str(i+1) +  "] = " + str(x[i]))


    if(opcao2 == 2):
            #-------------------------------criação das entradas
        a = np.array([]) #Arrays Unidimensionais
        b = np.array([])
        c = np.array([])
        d = np.array([])
        print()
        print("Escreva o tamanho da matriz quadrada")
        n = int(input())

        #criando valores para a[]
        for i_a in range (n-1): #de 1 a 19
            i_a_comp = i_a + 1
            a = np.append(a, ((2*i_a_comp - 1)/(4*i_a_comp))) #Adiciona elemento no fim da lista 

        a = np.append (a, (2*n - 1) / (2*n)) 


        #criando valores para b[]
        for i_b in range ((n)): #de 1 a 20
            i_b_comp = i_b + 1
            b = np.append(b, (2)) #Adiciona elemento no fim da lista 


        #criando valores para c[]
        for i_c in range ((n)): #de 1 a 20
            i_c_comp = i_c + 1
            c = np.append(c, (1 - (a[i_c]))) #Adiciona elemento no fim da lista 


        #criando os valores para d[]
        for i_d in range ((n)): #de 1 a 20
            i_d_comp = i_d + 1
            d = np.append( d, np.cos((2*(np.pi)*np.square(i_d_comp))/(np.square(n))))
            #criação de A[]
        A = np.zeros((n,n))
        A[0,n-1] = a[0]
        A[n-1,0] = c[n-1]
        for i in range (n):
            for j in range (n):
                if (i == j):
                    A[i, j] = b[i]
                if (i == j-1):
                    A[i,j] = c[i]
                if (i == j+1):
                    A[i,j] = a[i]
                    
        x = SistemasTridiagonaisCiclicos(A, d, n)
        print()
        print()
        print()
        print ("X: " )
        for i in range (n):
            print("X[" + str(i+1) +  "] = " + str(x[i]))


    if (opcao2 == 3):
        print("Digite o número de linhas da matriz quadrada")
        n = int(input())
        print("Escreva os elementos da matriz:")
        A = np.zeros((n,n))
        for i in range (n):
            print("Linha número " + str(i+1))
            for j in range (n):
                A[i,j] = input()
        print("Escreva o vetor d: ")
        d = np.zeros(n)
        for i in range (n):
            d[i] = int(input())
        x = SistemasTridiagonaisCiclicos(A, d, n)
        print()
        print()
        print()
        print ("X: " )
        for i in range (n):
            print("X[" + str(i+1) +  "] = " + str(x[i]))
