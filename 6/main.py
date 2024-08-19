from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset Iris
print("Cargando el dataset Iris...")
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset Iris cargado.")

# Dividir el dataset en datos de entrenamiento y prueba
print("Dividiendo el dataset en datos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Datos de entrenamiento: {X_train.shape[0]} muestras.")
print(f"Datos de prueba: {X_test.shape[0]} muestras.")

# Entrenar un modelo Random Forest
print("Entrenando el modelo Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Modelo entrenado.")

# Predecir las etiquetas de los datos de prueba
print("Prediciendo las etiquetas de los datos de prueba...")
y_pred = clf.predict(X_test)
print("Predicciones completadas.")

# Evaluar la efectividad del modelo
print("Evaluando la efectividad del modelo...")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
