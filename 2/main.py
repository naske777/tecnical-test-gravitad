import numpy as np

array = np.random.rand(5, 5)

suma_total = np.sum(array)

promedio_filas = np.mean(array, axis=1)

promedio_columnas = np.mean(array, axis=0)

valor_maximo = np.max(array)

valor_minimo = np.min(array)

print("Array:\n", array)
print("Suma total de los elementos:", suma_total)
print("Promedio de cada fila:", promedio_filas)
print("Promedio de cada columna:", promedio_columnas)
print("Valor máximo del array:", valor_maximo)
print("Valor mínimo del array:", valor_minimo)