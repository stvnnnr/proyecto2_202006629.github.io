# Manual de Usuario del Proyecto de Machine Learning en TytusJS

Este manual explica el uso de la aplicación para cada algoritmo de Machine Learning disponible en TytusJS. A continuación, se detallan los pasos para el ingreso de datos, entrenamiento y visualización de resultados específicos para cada modelo.

---

## 1. Regresión Lineal

### Paso 1: Cargar Datos
- Haz clic en **Cargar CSV** y selecciona tu archivo CSV.
- Verifica que los datos aparezcan en la tabla.
![Texto alternativo](images/1.png)


### Paso 2: Selección de Columnas
- En el menú **Algoritmo**, elige **Regresión Lineal**.
- Selecciona la columna **X** (variable independiente) y **Y** (variable dependiente).
![Texto alternativo](images/2.png)

### Paso 3: Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- La aplicación generará una **gráfica de regresión lineal** que muestra la relación entre los datos reales y los valores predichos.
![Texto alternativo](images/3.png)

---

## 2. Regresión Polinómica

### Paso 1: Cargar Datos
- Carga el archivo CSV siguiendo el mismo proceso que en la regresión lineal.
![Texto alternativo](images/4.png)

### Paso 2: Selección de Columnas y Grado del Polinomio
- Selecciona **Regresión Polinómica** en el menú **Algoritmo**.
- Selecciona las columnas **X** e **Y**.
- Especifica el **grado** del polinomio en el campo correspondiente.
![Texto alternativo](images/5.png)

### Paso 3: Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- La aplicación generará una **gráfica polinómica** y mostrará el coeficiente \( R^2 \) para evaluar el ajuste del modelo.
![Texto alternativo](images/6.png)

---

## 3. Árbol de Decisión

### Paso 1: Cargar Datos
- Carga el archivo CSV y verifica que los datos sean visibles.
![Texto alternativo](images/7.png)

### Paso 2: Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- La aplicación mostrará el **gráfico del árbol de decisión** visualizando cada nodo y la lógica de decisión.
![Texto alternativo](images/8.png)


---

## 4. Naive Bayes

### Paso 1: Cargar Datos
- Carga el archivo CSV en la aplicación.
![Texto alternativo](images/9.png)

### Paso 2: Selección de Columnas
- Rellena los datos necesarios solicitados por el **Algoritmo**.
![Texto alternativo](images/10.png)

### Paso 3: Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- La aplicación mostrará la **clase predicha** y la **probabilidad** calculada para esa clase.
![Texto alternativo](images/11.png)

---

## 5. Red Neuronal

### Paso 1: Cargar Datos
- Carga el archivo CSV en la aplicación.
![Texto alternativo](images/12.png)

### Paso 2: Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- Visualiza las **predicciones**, los **pesos** y **sesgos** de cada capa, y gráficos de barras que muestran los valores ajustados por la red.
![Texto alternativo](images/13.png)
![Texto alternativo](images/14.png)
---

## 6. K-Means

### Paso 1: Cargar Datos
- Carga el archivo CSV en la aplicación.
![Texto alternativo](images/15.png)

### Paso 2:Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- La aplicación mostrará un **gráfico de dispersión** con los puntos de datos agrupados por color y los centroides de cada cluster.
![Texto alternativo](images/16.png)

---

## 7. K-Vecinos Más Cercanos (KNN)

### Paso 1: Cargar Datos
- Carga el archivo CSV en la aplicación.
![Texto alternativo](images/17.png)

### Paso 2: Entrenamiento, Predicción y Visualización
- Haz clic en **Entrenar, Predecir y graficar**.
- La aplicación mostrará la **clase predicha** basada en los vecinos más cercanos a los valores de entrada proporcionados.
![Texto alternativo](images/18.png)

---