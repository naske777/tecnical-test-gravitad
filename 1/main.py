import pandas as pd
import re

# Leer el archivo CSV
df = pd.read_csv('WMT.csv')

# Mostrar las primeras filas del DataFrame
print(df.head())

# Calcular la media de la columna 'Close'
mean_value = df['Close'].mean()
print(f"Media de la columna 'Close': {mean_value}")

# Filtrar las filas según una condición relevante (por ejemplo, 'Close > 21')
filtered_df = df[df['Close'] > 21]

# Guardar el DataFrame filtrado en un archivo JSON con caracteres ASCII desactivados
filtered_df.to_json('filtered_WMT.json', orient='records', indent=4, date_format='iso', force_ascii=False)

# Leer el contenido del archivo JSON
with open('filtered_WMT.json', 'r', encoding='utf-8') as file:
    json_str = file.read()

# Reemplazar las barras invertidas en las fechas
json_str = re.sub(r'\\/', '/', json_str)

# Guardar el contenido modificado de nuevo en el archivo JSON
with open('filtered_WMT.json', 'w', encoding='utf-8') as file:
    file.write(json_str)