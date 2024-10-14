import random

def marche(n, p=0.5):
    x = [0]
    for i in range(n):
#        X = 2*random.randint(0,1)-1
        X = 2*random.binomialvariate(n=1, p=p)-1
        x.append(x[-1] + X)
    return x

def marche_cercle(n, p=0.5, M=10):
    x = [0]
    for i in range(n):
        X = 2*random.binomialvariate(n=1, p=p)-1
        xn = (x[-1] + X + M)%(2*M)-M
        x.append(xn)
    return x
