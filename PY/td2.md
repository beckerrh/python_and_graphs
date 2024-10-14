**Exercice 2**

Il y a plusieurs façon d'obteint les digits. Ici, on convertir d'abord en *str*.
- On utilise la comprehension de list: au lieu d'écrire une boucle et insérer on utilise une boucle dans la construction de la liste


```python
import random

a = random.randint(1000,9999)
d = [int(i) for i in str(a)]
""" équivalent à
d = []
for i in str(a):
    d.append(int(i)
"""
print(f"{a=} {d=}")
```

    a=4733 d=[4, 7, 3, 3]


On va donc crééer une liste avec 50 nombre aléatoires.


```python
na = [random.randint(-20,20) for i in range(50)]
print(na)    
```

    [-6, -12, 19, 15, 8, 17, 10, -10, -14, 1, 5, -19, -4, 0, 7, -13, 7, 1, -14, 15, -8, -9, -5, -4, -16, 5, 1, 10, 4, 10, 18, 1, -8, -17, -7, -16, -18, 8, -20, -8, 17, -20, -13, -8, -1, 9, -1, -20, 7, -2]


Combien de nombres difféernts ?


```python
print(len(set(na)))
```

    29


Pour la suite, nos 'expériences', on va d'abord mettre notre code en une fonction.
Ensuite on calculer les espérances.


```python
def nbdiff():
    na = [random.randint(-20,20) for i in range(50)]
    return len(set(na))

for n in [10,20,40,80]:
    vals = [nbdiff() for i in range(n)]
    esp = sum(vals)/n
    print(f"{n=} {esp=}")
```

    n=10 esp=29.2
    n=20 esp=29.35
    n=40 esp=29.2
    n=80 esp=29.25


**Exercice 3**

On rappelle que dans l'exercice 6 du TD 1 nous avons créé un fichier 'poker.py' qui contient les fonctions pour distribuer les cartes. 


```python
import poker
jeux = poker.distribuer_cartes(3)
print(jeux)
jeu = jeux[0]
print(jeu)
coul = [c[0] for c in jeu]
val = [c[1] for c in jeu]
print(val)
```

    jeu=['CA', 'CK', 'CD', 'CJ', 'C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'PA', 'PK', 'PD', 'PJ', 'P0', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'TA', 'TK', 'TD', 'TJ', 'T0', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'KA', 'KK', 'KD', 'KJ', 'K0', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9']
    [['C5', 'PD', 'C4', 'P3', 'P2'], ['T7', 'P0', 'KA', 'PA', 'K0'], ['TD', 'T8', 'K8', 'PK', 'C0']]
    ['C5', 'PD', 'C4', 'P3', 'P2']
    ['5', 'D', '4', '3', '2']


Maintenant on écrit une fonction pour trouver les fréquences (paires, brelans, carrés). Il y a différentes façon de faire cela. Ici, on utiliser un dictionnaire pour conter. 


```python
def frequences(valeurs):
    d = {}
    for v in valeurs:
        if v in d.keys():
            d[v] += 1
        else:
            d[v] = 1
    return d

for jeu in jeux:
    val = [c[1] for c in jeu] 
    print(frequences(val))
```

    {'5': 1, 'D': 1, '4': 1, '3': 1, '2': 1}
    {'7': 1, '0': 2, 'A': 2}
    {'D': 1, '8': 2, 'K': 1, '0': 1}


Maintenant on peut écrire un fonction qui évalue un jeu donné. Encore une fois, il y a maintes faàon de faire. Ici, on va faire un peu plus que ce qui est demandé. Dans le cas d'un brelan, on distingue le cas dans lequel on a aussi une paire ("full house"). Et puis, on détermine les différents types de cartes. Pour cela, on crée un autre *dictionnaire*, qui donne pour chaque fréquence (entre 1 et 4) un *set* contenant les valeurs.

Quelques explications techniques:
- on n'a pas besoin d'utiliser **elif** et **else**, car la fonction quitte avec **return**


```python
def evaluer(jeu):
    val = [c[1] for c in jeu]
    d = frequences(val)
    e = {1:[], 2:[], 3:[], 4:[]}
    for k,v in d.items():
        e[v].append(k)
    if len(e[4]):
        return "Carré " + "de " + e[4][0]
    if len(e[3]):
        if len(e[2]):
            return "FullHouse " + "de " + e[3][0] + e[2][0]
        else:
            return "Brelan " + "de " + e[3][0]
    if len(e[2])==2:
        return "TwoPairs " + "de " + e[2][0] + e[2][1]
    if len(e[2])==1:
        return "Pair " + "de " + e[2][0]
    return "HighCard"

for jeu in jeux:
    print(f"valeur de {jeu} : {evaluer(jeu)}")
```

    valeur de ['C5', 'PD', 'C4', 'P3', 'P2'] : HighCard
    valeur de ['T7', 'P0', 'KA', 'PA', 'K0'] : TwoPairs de 0A
    valeur de ['TD', 'T8', 'K8', 'PK', 'C0'] : Pair de 8


**Exercice 5**

On a $x_0=0$ et ensuite $$x_{i+1}=x_i + X_i$$.
On utilise la formule du cours pour créer la marche aléatoire. Notre fonction prend deux arguments : le nombre de pas et la probabilité. 
- Pour permettre $p\ne0.5$ on utilise *random.binomialvariate* au lieu de *random.randint*.
- On utilise un argument avec valeur par défault, ce qui facilite l'utilisation


```python
import random

def marche(n, p=0.5):
    x = [0]
    for i in range(n):
#        X = 2*random.randint(0,1)-1
        X = 2*random.binomialvariate(n=1, p=p)-1
        x.append(x[-1] + X)
    return x

print(marche(100))
```

    [0, 1, 2, 1, 0, -1, 0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 2, 1, 0, 1, 2, 1, 2, 3, 4, 5, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 2, 1, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 5, 4, 3, 2, 1, 0, -1, 0, 1, 0, 1, 0, 1, 2, 1, 0, -1, 0, -1, -2, -3, -2, -3, -4, -3, -2, -1, 0, -1, -2, -1, -2, -1, 0, 1, 2, 3, 4, 5, 4, 5, 4]


Finalement, pour la variante, on utilise **%** (modulo). Il faut faire un peu attention : $\%n$ identifie $n$ avec $0$, alors on va d'abord décaler notre intervalle $[-M,M[$ à $[0,2M[$. Après le modulo on revient. 


```python
M = 10
def mod(x,M=10):
    return (x+M)%(2*M)-M
print(mod(10), mod(11), mod(-10), mod(-11))

```

    -10 -9 -10 9



```python
def marche_cercle(n, p=0.5, M=10):
    x = [0]
    for i in range(n):
        X = 2*random.binomialvariate(n=1, p=p)-1
        xn = (x[-1] + X + M)%(2*M)-M
        x.append(xn)
    return x
    
print(marche_cercle(100, p=0.8, M=5))
```

    [0, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -5, -4, -5, -4, -3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -5, -4, -3, -2, -1, 0, -1, 0, 1, 2, 3, 4, -5, 4, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -5, -4, -3, -2, -3, -4, -3, -2, -1, 0, 1, 0, -1, -2, -1, 0, -1, 0, -1, 0, 1, 2, 3, 4, -5, -4]



```python

```
