# Manual Técnico del Proyecto de Machine Learning en TytusJS

Este manual describe la estructura y funcionalidad de las principales funciones implementadas en el proyecto de Machine Learning utilizando TytusJS.

## 1. Estructura del Código

El proyecto está compuesto por funciones que manejan diversas tareas relacionadas con algoritmos de Machine Learning, incluyendo gráficos y predicciones. A continuación, se detallan las principales funciones y su funcionamiento.

---

## 2. Inicialización del Proyecto

**Función:** `init`

- **Descripción**: Esta función inicializa la aplicación, cargando los elementos y configuraciones necesarias para comenzar a trabajar con los datos y algoritmos de Machine Learning.

---

## 3. Cargar Datos CSV

**Función:** `loadCSV`

- **Descripción**: Permite cargar y procesar un archivo CSV. Separa los datos en encabezados y filas de datos.

- **Uso**:
  ```javascript
  loadCSV(csvFile, csvNumber); // Carga el archivo CSV y lo procesa para su uso en el modelo.
  ```

## 4. Procesar Datos del CSV

**Función:** `processCSV`

- **Descripción**: Divide el archivo CSV en filas y columnas, almacena encabezados y datos, y actualiza los selectores de columna para el usuario.

- **Uso**:
  ```javascript
  processCSV(csvFileContent, 1); // Procesa el contenido del archivo CSV con ID 1.
  ```

## 5. Mostrar Gráfica de Regresión Lineal

**Función:** `showLinearGraph`

- **Descripción**: Genera un gráfico de línea para mostrar las predicciones realizadas por el modelo de regresión lineal en comparación con los valores reales.

- **Uso**:
  ```javascript
  const xValues = [1, 2, 3, 4];
  const yValues = [2, 4, 6, 8];
  showLinearGraph(xValues, yValues); // Muestra el gráfico de regresión lineal
  ```

## 6. Mostrar Gráfica de Regresión Polinómica

**Función:** `showPolynomialGraph`

- **Descripción**: Genera un gráfico de línea que compara los valores reales con los valores predichos por la regresión polinómica. Calcula y muestra el valor \( R^2 \) para evaluar el modelo.

- **Uso**:
  ```javascript
  const degree = 2; // Grado del polinomio
  polynomialModel.fit(xValues, yValues, degree);
  const predictions = polynomialModel.predict(xValues);
  showPolynomialGraph(xValues, yValues); // Muestra el gráfico polinómico con \( R^2 \)
  ```

## 7. Mostrar Gráfica para Árbol de Decisión

**Función:** `showDecisionTreeGraph`

- **Descripción**: Muestra una visualización del árbol de decisión en un diseño jerárquico basado en el formato DOT.

- **Uso**:
  ```javascript
  const dotStr = decisionTreeModel.generateDotString(root);
  showDecisionTreeGraph(dotStr); // Visualiza el árbol de decisión
  ```

## 8. Mostrar Resultado de la Predicción Naive Bayes

**Función:** `showNaiveBayesPrediction`

- **Descripción**: Muestra el resultado de una predicción realizada por el modelo Naive Bayes, indicando la clase y la probabilidad de la predicción.

- **Uso**:
  ```javascript
  const prediction = model.predict(inputValues);
  showNaiveBayesPrediction(prediction); // Muestra el resultado de la predicción Naive Bayes
  ```

## 9. Visualizar Predicciones de Red Neuronal

**Función:** `showNeuralNetworkPredictions, showNeuralNetworkWeights, showNeuralNetworkBiases`

- **Descripción**: Permite ver las predicciones, pesos y sesgos de la red neuronal entrenada.

- **Uso**:

  ```javascript
  const predictions = nn.predict(data); // Predicciones de la red
  showNeuralNetworkPredictions(predictions);

  const weights = nn.layerLink[0].obtener_Weights().data; // Pesos de la primera capa
  showNeuralNetworkWeights(weights);

  const biases = nn.layerLink[0].obtener_Bias().data; // Sesgos de la primera capa
  showNeuralNetworkBiases(biases);
  ```

## 10. Gráfica de Pesos y Sesgos para Redes Neuronales

**Función:** `drawWeightsBiasesChart`

- **Descripción**: Genera un gráfico de barras que compara los pesos promedio y sesgos por capa de la red neuronal.

- **Uso**:
  ```javascript
  const weights = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ]; // Ejemplo de pesos
  const biases = [[0.1], [0.2]]; // Ejemplo de sesgos
  drawWeightsBiasesChart(weights, biases); // Muestra el gráfico de pesos y sesgos
  ```

## 11. Clustering con K-Means

**Función:** `showKMeansClusters`

- **Descripción**: Realiza el clustering de datos utilizando el algoritmo K-Means y muestra los clusters generados en un gráfico de dispersión. Cada cluster se representa con un color diferente, y también muestra los centroides de cada grupo.

- **Parámetros**:
  - `k` - Número de clusters.
  - `xValues` - Coordenadas X de los puntos de datos.
  - `yValues` - Coordenadas Y de los puntos de datos.

- **Uso**:
  ```javascript
  const k = 3; // Número de clusters
  const xValues = [1, 2, 3, 4, 5];
  const yValues = [5, 4, 3, 2, 1];
  showKMeansClusters(xValues, yValues, k); // Muestra el gráfico de clusters
  ```


## 12. Clasificación con K-Vecinos Más Cercanos (KNN)

**Función:** `showKNNPrediction`

- **Descripción**: Realiza la predicción de una clase usando el algoritmo K-Vecinos Más Cercanos (KNN) y muestra los resultados en pantalla. Se basa en la distancia entre el punto de datos y sus vecinos más cercanos para clasificarlo en una categoría.

- **Parámetros**:
   - `k`: Número de clusters.
   - `xValues`: Coordenadas X de los puntos de datos.
   - `yValues`: Coordenadas Y de los puntos de datos.


- **Uso**:
  ```javascript
    const k = 5; // Número de vecinos más cercanos
    const inputValues = [2.5, 3.5]; // Datos del punto a clasificar
    const trainingData = [
        { features: [1, 2], label: "Clase A" },
        { features: [3, 4], label: "Clase B" },
    ];
    const prediction = knnModel.predict(inputValues, trainingData, k);
    showKNNPrediction(prediction); // Muestra la clase predicha en la pantalla
  ```

## 13. Manejar el Entrenamiento y la Predicción

**Función:** `handleTrain`

- **Descripción**: Esta función coordina el entrenamiento y predicción de diversos algoritmos de Machine Learning (regresión lineal, polinómica, árboles de decisión, Naive Bayes, redes neuronales, K-Means, K-Vecinos Más Cercanos).

- **Uso**:

  ```javascript
  document.getElementById("trainButton").addEventListener("click", () => {
    const selectedAlgorithm = "linear_regression";
    const xIndex = 0;
    const yIndex = 1;

    const xValues = csvData1.map((row) => row[xIndex]);
    const yValues = csvData1.map((row) => row[yIndex]);

    handleTrain(selectedAlgorithm, xValues, yValues);
  });
  ```
