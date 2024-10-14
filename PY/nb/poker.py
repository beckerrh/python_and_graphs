import random

couleurs = 'CPTK'
valeurs = 'AKDJ023456789'

def creer_jeu():
    jeu = []
    for c in couleurs:
        for v in valeurs:
            jeu.append(c+v)
    random.shuffle(jeu)
    return jeu

def distribuer_cartes(njoueurs):
    jeu = creer_jeu()
    jeux_joueurs = []
    for i in range(njoueurs):
        jeux_joueurs.append([jeu.pop() for j in range(5)])
    return jeux_joueurs