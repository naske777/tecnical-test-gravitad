import pandas as pd

# Cargar el dataset en un DataFrame
df = pd.read_csv('WMT.csv')

# Imprimir las primeras 5 filas del DataFrame
print(df.head())

# Calcular la media de la columna 'Close'
mean_value = df['Close'].mean()
print(f"Media de la columna 'Close': {mean_value}")

# Filtrar las filas según una condición relevante (por ejemplo, 'Close > 21')
filtered_df = df[df['Close'] > 21]

# Guardar el DataFrame resultante en un nuevo archivo JSON
filtered_df.to_json('filtered_WMT.json', orient='records', lines=True)