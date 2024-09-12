# Summary
These notes are for absolute beginners in **python** and **graphs**.

We will look at:
- list (basic usage)
- a graph represented by an edge list
- a graph represented by an adjacency list

# List (basic usage)
- symbol [], elements separated by comma
- ordered
- heterogenuous ('anything' can be an element)
first things to know:
- access operator []
- many functions like (*pop* *append* ...)


```python
l = [12, 'abra', [2, 3.1]]
print(l[0], l[-1], l[-1][1])
```

    12 [2, 3.1] 3.1



```python
print(l.pop(), l)
```

    [2, 3.1] [12, 'abra']



```python
print(l.pop(0), l)
```

    12 ['abra']



```python

```
