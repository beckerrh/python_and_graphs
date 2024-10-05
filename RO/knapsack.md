We consider the knapsack problem: given $n\in\mathbb N$, $c\in\mathbb N^n$, $w\in\mathbb N^n$, and $W\in \mathbb N$ representing the number, utility, and weight objects, and the size of the backpack, we wish to do our best:
$$
\left\{
\begin{aligned}
&\max c^T x\\
&w^T x \le W\\
&x\in\{0,1\}^n
\end{aligned}
\right.
$$
Here $\xi_i=1$ and $x_i=0$ mean that object $i$ is selected or not. We can suppose $W<\sum\limits_{i=1}^nw_i$, since otherwise we can take all.

We first look at the relaxed problem:
$$
\left\{
\begin{aligned}
&\max c^T x\\
&w^T x \le W\\
&x \le e \qquad (e = (1,\cdots,1)^T))\\
&x\in\mathbb R_+^n
\end{aligned}
\right.
$$

The dual of this problem is:
$$
\left\{
\begin{aligned}
&\min z W +  e^T y\\
& y + z w \ge c\\
&z\ge0,\quad y\ge0
\end{aligned}
\right.
$$
As we will see, we can nearly solve this problem 'by hand'.

Let $x^*$ and $(z^*,y^*)$ be a pair of optimal solutions. By duality we have
$$
z^* W + e^T y^* = c^T x^*
$$
and complementarity
$$
z^*(W - w^T x^*) = 0, \quad (1-x_i^*)y_i^*=0.
$$
If $z^*>0$, we have $y^*=c$ and then $x^*=e$, which is not possoble by assumption (our backpack is too small to take all).

Now we see from the dual problem that $y^*_i\ge0$ and $y^*_i\ge c_i - z^* w_i$. Then
$$
y^*_i = \max\{0, c_i - z^* w_i\} = w_i \max\{0,\gamma_i - z^*\},\quad \gamma_i := \frac{c_i}{w_i},
$$
since otherwise it cannot be optimal. So we get $y^*$, if we know $z^*$! Similarily, we see that $z^*=\gamma_{i^*}$ for some $1\le i^*\le n$. 
To makes things simple, we suppose that data are already order such that
$$
\gamma_{1}\ge \cdots \ge \gamma_{n}.
$$
Then we have
$$
y^*_i=
\begin{cases}
w_i (\gamma_i - \gamma_{i^*})&\quad i < i^*\\
0 &\quad i \ge i^*.
\end{cases}
$$
So we get with strong duality 
$$
z^*W  = c^T x^* - e^T y^* = \sum_{i=1}^{i^*-1}c_i + \sum_{i=i^*}^{n}c_ix^*_{i} - \sum_{i=1}^{i^*-1}w_i (\gamma_i - z^*)
=  c_{i^*}x^*_{i^*} + z^*\sum_{i=1}^{i^*-1}w_i
$$
Therefore $i^*$ is defined by
$$
\sum_{i=1}^{i^*-1} w_i \le W \quad\text{and}\quad \sum_{i=1}^{i^*} w_i>W
$$
and
$$
x^*_i=
\begin{cases}
1 &\quad i < i^*\\
\frac{1}{w_{i^*}}\left(W - \sum\limits_{i=1}^{i^*-1} w_i\right) &\quad i =i^*\\
0 &\quad i > i^*.
\end{cases}
$$
This is natural we take the objects with best ratios 'utility/weight'.

Let us put this into python:


```python
import numpy as np

def knapsack(c, w, W):
    c = c.astype(float)
    gamma = c/w
    pi = np.argsort(gamma)[::-1]
    total = 0
    x = np.zeros_like(c)
    for i in pi:
        if total + w[i] > W:
            x[i] = (W-total)/w[i]
            break
        total += w[i]
        x[i] = 1
    return x


c, w, W = np.array([5,3,6,6,2]),np.array([5,4,7,6,2]), 15
x = knapsack(c, w, W)
print(f"{x=} {x.dot(c)=} {x.dot(w)=}")

```

    x=array([1.        , 0.        , 0.28571429, 1.        , 1.        ]) x.dot(c)=np.float64(14.714285714285715) x.dot(w)=np.float64(15.0)


As expected, we do not get a binary solution. So we can 'branch and bound'. The two possibilites are to fix $x_3=0$ or $x_3=1$.
In order to do so, we modify our function, such it takes into account the fixed variables, either equal to zero or equal to one.


```python
def knapsack(c, w, W, Izero=[], Ione=[]):
    c = c.astype(float)
    gamma = c/w
    pi = np.argsort(gamma)[::-1]
    total = 0
    x = np.zeros_like(c)
    x[Ione] = 1
    W -= w[Ione].sum()
    for i in pi:
        if i in Ione or i in Izero:
            continue
        if total + w[i] > W:
            x[i] = (W-total)/w[i]
            break
        total += w[i]
        x[i] = 1
    binary = '***' if np.all(x*(1-x)==0) else '---' 
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(f"{str(Izero):10} {str(Ione):10} {binary} {x=} {x.dot(c)=:6.3f} {x.dot(w)=:6.3f}")
    return x
c, w, W = np.array([5,3,6,6,2]),np.array([5,4,7,6,2]), 15
x = knapsack(c, w, W)
print("First branch")
x = knapsack(c, w, W, Ione=[2])
x = knapsack(c, w, W, Izero=[2])
print("Second branch")
x = knapsack(c, w, W, Izero=[1,2])
x = knapsack(c, w, W, Izero=[2], Ione=[1])
print("Third branch")
x = knapsack(c, w, W, Izero=[0,2], Ione=[1])
x = knapsack(c, w, W, Izero=[2], Ione=[0,1])
print("Forth branch")
x = knapsack(c, w, W, Izero=[2,3], Ione=[0,1])
x = knapsack(c, w, W, Izero=[2], Ione=[0,1,3])

```

    []         []         --- x=array([ 1.000,  0.000,  0.286,  1.000,  1.000]) x.dot(c)=14.714 x.dot(w)=15.000
    First branch
    []         [2]        *** x=array([ 0.000,  0.000,  1.000,  1.000,  1.000]) x.dot(c)=14.000 x.dot(w)=15.000
    [2]        []         --- x=array([ 1.000,  0.500,  0.000,  1.000,  1.000]) x.dot(c)=14.500 x.dot(w)=15.000
    Second branch
    [1, 2]     []         *** x=array([ 1.000,  0.000,  0.000,  1.000,  1.000]) x.dot(c)=13.000 x.dot(w)=13.000
    [2]        [1]        --- x=array([ 0.600,  1.000,  0.000,  1.000,  1.000]) x.dot(c)=14.000 x.dot(w)=15.000
    Third branch
    [0, 2]     [1]        *** x=array([ 0.000,  1.000,  0.000,  1.000,  1.000]) x.dot(c)=11.000 x.dot(w)=12.000
    [2]        [0, 1]     --- x=array([ 1.000,  1.000,  0.000,  0.667,  1.000]) x.dot(c)=14.000 x.dot(w)=15.000
    Forth branch
    [2, 3]     [0, 1]     *** x=array([ 1.000,  1.000,  0.000,  0.000,  1.000]) x.dot(c)=10.000 x.dot(w)=11.000
    [2]        [0, 1, 3]  *** x=array([ 1.000,  1.000,  0.000,  1.000,  0.000]) x.dot(c)=14.000 x.dot(w)=15.000


That makes a lot of iterations (nearly trail and error, butter better organized!). We got two solutions, all with value 14:
$$
    \begin{aligned}
x = (0, 0, 1, 1, 1)\\
x = (1, 1, 0, 1, 0)
\end{aligned}
$$
So what do you pack?


```python

```
