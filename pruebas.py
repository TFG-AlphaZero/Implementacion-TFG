#Archivo para probar mierdas varias.
import numpy as np

lista = np.array([1,2,3,4,5])
res = np.argmax(np.vectorize(lambda i : -np.log(i))(lista))

#print(res)

counter = 4
counter -= 1
print(counter)