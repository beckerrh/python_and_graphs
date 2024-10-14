**Exercice 1**


```python
import marche_alea

x = marche_alea.marche(100)
d = {}
for xe in x:
    if xe in d.keys():
        d[xe] += 1
    else:
        d[xe] = 1

print(f"{d=} \n{sorted(d.items())=}")
```

    d={0: 3, -1: 8, -2: 17, -3: 23, -4: 18, -5: 13, -6: 11, -7: 6, -8: 2} 
    sorted(d.items())=[(-8, 2), (-7, 6), (-6, 11), (-5, 13), (-4, 18), (-3, 23), (-2, 17), (-1, 8), (0, 3)]


**Exercice 3**
On peut utiliser l'exercice 1. **numpy** nous donne une autre possibilité. On peut donc comparer.
Quelques remarques :
- *sorted(d.items())* est une **list**, on convertit on **dict**
- la fonction **np.histogram()** prend un argument 'bins' (les paliers), qui est en général le nombre de 'bins' souhaités. Cela suffit en général, mais ici on veut avoir avec le premier résultat.
- la fonction **np.histogram()** rend deux arguments : les fréquences et les intervalles 


```python
import marche_alea
import numpy as np

def hist_dict(x):
    d = {}
    for xe in x:
        if xe in d.keys():
            d[xe] += 1
        else:
            d[xe] = 1
    return dict(sorted(d.items()))

def hist_np(x, bins):
    h = np.histogram(x, bins=bins)
    return h

x = marche_alea.marche(100)
d = hist_dict(x)
print(f"{d=}")
h = hist_np(x, len(d.keys()))
print(f"{d=}\n{h=}")
```

    d={-6: 2, -5: 6, -4: 8, -3: 6, -2: 9, -1: 15, 0: 19, 1: 18, 2: 11, 3: 5, 4: 2}
    d={-6: 2, -5: 6, -4: 8, -3: 6, -2: 9, -1: 15, 0: 19, 1: 18, 2: 11, 3: 5, 4: 2}
    h=(array([ 2,  6,  8,  6,  9, 15, 19, 18, 11,  5,  2]), array([-6.        , -5.09090909, -4.18181818, -3.27272727, -2.36363636,
           -1.45454545, -0.54545455,  0.36363636,  1.27272727,  2.18181818,
            3.09090909,  4.        ]))


**Exercice 4**


```python
def graphe_cours():
    E = [(0,1), (2,1), (3,2), (4,3), (0,4), (1,4), (2,0), (4,2)]
    return 5, E

def graphe_complet(n):
    E = []
    for i in range(n):
        for j in range(n):
            if i != j:
                E.append((i,j))
                E.append((j,i))
    return n, E

def construit_graphe_nonoriente(G):
    n, E = G[0], G[1]
    E2 = []
    for e in E:
        E2.append((e[0], e[1]))
        E2.append((e[1], e[0]))
    return n, E2

G = graphe_cours()
G2 = construit_graphe_nonoriente(G)
print(f"{G=}\n{G2=}")
```

    G=(5, [(0, 1), (2, 1), (3, 2), (4, 3), (0, 4), (1, 4), (2, 0), (4, 2)])
    G2=(5, [(0, 1), (1, 0), (2, 1), (1, 2), (3, 2), (2, 3), (4, 3), (3, 4), (0, 4), (4, 0), (1, 4), (4, 1), (2, 0), (0, 2), (4, 2), (2, 4)])



```python
def liste_adjacence(G):
    n, E = G[0], G[1]
    a = [[] for i in range(n)]
    for e in E:
        a[e[0]].append(e[1])
    return a

G = graphe_cours()
A = liste_adjacence(G)
print(f"{G=}\n{A=}")
```

    G=(5, [(0, 1), (2, 1), (3, 2), (4, 3), (0, 4), (1, 4), (2, 0), (4, 2)])
    A=[[1, 4], [4], [1, 0], [2], [3, 2]]



```python

```
