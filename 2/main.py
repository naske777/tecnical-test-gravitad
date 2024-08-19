import numpy as np

# Crear un array de tamaño 5x5 con valores aleatorios
array = np.random.rand(5, 5)

# Calcular la suma de todos los elementos del array
suma_total = np.sum(array)

# Calcular el promedio de cada fila
promedio_filas = np.mean(array, axis=1)

# Calcular el promedio de cada columna
promedio_columnas = np.mean(array, axis=0)

# Encontrar el valor máximo del array
valor_maximo = np.max(array)

# Encontrar el valor mínimo del array
valor_minimo = np.min(array)

# Imprimir los resultados
print("Array:\n", array)
print("Suma total de los elementos:", suma_total)
print("Promedio de cada fila:", promedio_filas)
print("Promedio de cada columna:", promedio_columnas)
print("Valor máximo del array:", valor_maximo)
print("Valor mínimo del array:", valor_minimo)