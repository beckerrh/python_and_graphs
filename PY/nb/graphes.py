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

def liste_adjacence(G):
    n, E = G[0], G[1]
    a = [[] for i in range(n)]
    for e in E:
        a[e[0]].append(e[1])
    return a

def visu_graphs_cercle(G, r=2):
    n, E = G
    delta = 2*math.pi/n
    t = [i*delta for i in range(n)]
    x, y = [r*math.cos(te) for te in t], [r*math.sin(te) for te in t]
    for i, (xe, ye) in enumerate(zip(x,y)):
        plt.plot(xe, ye, 'x', label='v'+str(i))
    for e in E:
        xe, ye = [x[e[0]], x[e[1]]], [y[e[0]], y[e[1]]]
        plt.plot(xe, ye, 'k', lw=1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    G = graphe_cours()
    visu_graphs_cercle(G)