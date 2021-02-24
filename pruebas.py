#Archivo para probar mierdas varias.
import numpy as np

from tfg.strategies import MonteCarloTreeNode

lista = np.array([40, 10, 5, 10, 15, 320])
res = np.argmax(np.vectorize(lambda i : -np.log(i))(lista))

#print(res)
nodes = [MonteCarloTreeNode(None, None) for i in range(6)]
for i in range(6):
    nodes[i].visit_count = lista[i]

t = 1
fun1 = lambda i : i.visit_count**(1/t)
fun2 = np.vectorize(fun1)

sum = np.sum(fun2(nodes))

pi = [fun1(node) / sum for node in nodes]
print(pi)

