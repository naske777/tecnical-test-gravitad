import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

# Generar un conjunto de datos sintéticos
print("Generando un conjunto de datos sintéticos...")
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=0)
print("Datos sintéticos generados.")

# Dividir el conjunto de datos en datos de entrenamiento y prueba
print("Dividiendo el conjunto de datos en datos de entrenamiento y prueba...")
X_train = X[:200]
X_test = X[200:]
print(f"Datos de entrenamiento: {X_train.shape[0]} muestras.")
print(f"Datos de prueba: {X_test.shape[0]} muestras.")

# Genera algunas anomalías regulares
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# Entrenar un modelo One-Class SVM
print("Entrenando el modelo One-Class SVM...")
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
print("Modelo entrenado.")

# Predecir las etiquetas de los datos de entrenamiento
print("Prediciendo las etiquetas de los datos de entrenamiento...")
y_pred_train = clf.predict(X_train)
print("Predicciones de entrenamiento completadas.")
print(f"Etiquetas predichas para los datos de entrenamiento: {np.unique(y_pred_train, return_counts=True)}")

# Predecir las etiquetas de los datos de prueba
print("Prediciendo las etiquetas de los datos de prueba...")
y_pred_test = clf.predict(X_test)
print("Predicciones de prueba completadas.")
print(f"Etiquetas predichas para los datos de prueba: {np.unique(y_pred_test, return_counts=True)}")

# Evaluar la efectividad del modelo
print("Evaluando la efectividad del modelo...")
y_true_train = np.ones_like(y_pred_train)  # Asumimos que todos los datos de entrenamiento son normales
y_true_test = np.ones_like(y_pred_test)  # Asumimos que todos los datos de prueba son normales

# Calcular y mostrar el reporte de clasificación y la matriz de confusión
print("Reporte de clasificación para los datos de entrenamiento:")
print(classification_report(y_true_train, y_pred_train))
print("Matriz de confusión para los datos de entrenamiento:")
print(confusion_matrix(y_true_train, y_pred_train))

print("Reporte de clasificación para los datos de prueba:")
print(classification_report(y_true_test, y_pred_test))
print("Matriz de confusión para los datos de prueba:")
print(confusion_matrix(y_true_test, y_pred_test))

# Visualizar los resultados
print("Visualizando los resultados...")
plt.title("Detección de anomalías con One-Class SVM")
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='blueviolet', s=s, edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2],
           ["superficie de decisión", "observaciones de entrenamiento",
            "anomalías"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.savefig("one_class_svm_anomalies.png")
print("Visualización guardada en 'one_class_svm_anomalies.png'.")