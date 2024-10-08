**Exercice 1**

- il ne faut oublier de convertir le retour (qui est toujours un **str**)
- pour accéder à $\pi$ on peut utiliser la librairie *math*
- la division '/' est touours convertie en float; pour la divsion entière on utilise '//' et '%' pour le reste de la division. Ainsi euclide s'écrit comme $$ n = m//p + m\%p$$ 



```python
import math

res = input("Donnez-moi le rayon svp !")
peri = 2*math.pi*float(res)
print(f"Le périmètre est {peri}")
```

Pour analyser si un nombre est premier ... il suffit d'essayer. Comment peut-on améliorer le code suivant ?


```python
def est_premier(n):
    for k in range(2,n):
        if n%k == 0:
            return False, k
    return True, 0

res = input("Donnez-moi un entier svp !")
n = int(res)
prem, k = est_premier(n)
if prem:
    print(f"Le nombre {n} est premier")
else:
   print(f"Le nombre {n} n'est pas premier, car divisible par {k}") 
```

**Exercice 2**


```python
import string
a = string.ascii_lowercase
print(a[-1])
print(a[::-1])
print(a[4:8])
print(a[-6:-2])
print(a[-5:])
print(a[5::2])
print(a[-2:2:-3])
```

    z
    zyxwvutsrqponmlkjihgfedcba
    efgh
    uvwx
    vwxyz
    fhjlnprtvxz
    yvspmjgd


**Exercice 5**

Une liste est ordonnée et peut contenir des doublons. 
Une façon de répondre est de crééer une nouvelle liste avec les élements uniques.

Prenons cet exemple pour montrer comment on peut développer une fonction.
- d'abord on prend un exemple


```python
l = [1, 3, 7 , 'a', 3, 'b']
lu = []
for le in l:
    if not le in lu:
        lu.append(le)
print(f"{lu=} {len(l)=} {len(lu)=}")
```

    lu=[1, 3, 7, 'a', 'b'] len(l)=6 len(lu)=5



```python
def is_set(l):
    lu = []
    for le in l:
        if le not in lu:
            lu.append(le)
    return len(l)==len(lu), lu

#test
print(is_set(l))
```

    (False, [1, 3, 7, 'a', 'b'])


Maintenant on va écrire une fonction pour la différence. On utilise la fonction précédente, mais nous n'avons besoin que la deuxième valeur de retour.


```python
def difference_listes(l1, l2):
    l = []
    for el in l1:
        if el not in l2:
            l.append(el)
    return is_set(l)[1]

#test
print(difference_listes([1,2,3,1,5,7,7], [1,2,3]))
```

    [5, 7]


En utilsant les *set* c'est bien plus facile !


```python
l1, l2 = [1,2,3,1,5,7,7], [1,2,3]
print(f"l1 est unique ? {len(l1)==len(set(l1))}")
print(f"difference : {set(l1).difference(l2)}")
```

    l1 est unique ? False
    difference : {5, 7}


**Exercice 6 (Poker 1)** 

On utilise
- pour le couleurs : 'C': coeur, 'P': pique, 'T': trèfles, 'K': caro
- pour les valeurs : 'AKDJ023456789' : '0' pour 10, 'A' pour l'as

Ensuite on crée le jeu de carte, et puis on le distribue avec *shuffle*.


```python
import random

couleurs = 'CPTK'
valeurs = 'AKDJ023456789'

jeu = []
for c in couleurs:
    for v in valeurs:
        jeu.append(c+v)
print(f"{jeu=}")
random.shuffle(jeu)

njoueurs = 4
jeux_joueurs = []
for i in range(njoueurs):
    jeux_joueurs.append([jeu.pop() for j in range(5)])

for i, jeu in enumerate(jeux_joueurs):
    print(f"joueurs {i} : {jeu}")
```

    jeu=['CA', 'CK', 'CD', 'CJ', 'C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'PA', 'PK', 'PD', 'PJ', 'P0', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'TA', 'TK', 'TD', 'TJ', 'T0', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'KA', 'KK', 'KD', 'KJ', 'K0', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9']
    joueurs 0 : ['C7', 'K4', 'P9', 'T0', 'K6']
    joueurs 1 : ['P7', 'KD', 'P0', 'K9', 'P5']
    joueurs 2 : ['CK', 'P4', 'TJ', 'T8', 'K7']
    joueurs 3 : ['C8', 'CA', 'T2', 'T4', 'KK']


**Exercice 7**
Un dictionnaire correspond à une application injective, si deux clés différentes sont associées à deux valeurs différentes. Il suffit de voir, si le valeurs sont deux à deux différentes. On peut alors faire comme dans l'exercice 5.



```python
def dict_injectif(d):
    return len(set(d.values()))==len(d.values())

d = {}
d[1] = 'a'
d[2] = 'c'
# on aurait pu écrire d = {1:'a', 2:'c'}
print(dict_injectif(d))
d[3] = 'a'
print(dict_injectif(d))
```

    True
    False


On construit l'application inverse en inversant clé et valeur. On rajoute un petit test pour savoir, si ça marche.


```python
def dict_inverse(d):
    di = {}
    for k,v in d.items():
        di[v] = k
    if len(di) != len(d):
        print(f"dictionary not injectif : {d}")
    return di
print(dict_inverse(d))
print(dict_inverse({1:'a', 2:'c'}))
```

    dictionary not injectif : {1: 'a', 2: 'c', 3: 'a'}
    {'a': 3, 'c': 2}
    {'a': 1, 'c': 2}



```python

```
