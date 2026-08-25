# Clasificación Multiclase de Caracteres (notMNIST) — Regresión Logística One-vs-All

**Estudiante:** Vela Gonzales Alex Sander
**Laboratorio:** Laboratorio 5: Regresión Logística one vs all

## Descripción

Este repositorio contiene la implementación completa de un modelo de **regresión logística multiclase** (estrategia **One-vs-All**, programada desde cero con NumPy) para clasificar imágenes del dataset **notMNIST** en 10 clases (letras A a J).

- **Dataset elegido:** [notMNIST](https://www.kaggle.com/datasets/jwjohnson314/notmnist) (subconjunto `notMNIST_large`, ~529,000 imágenes originales de 28×28 px)
- **Muestra utilizada:** 60,000 imágenes balanceadas (6,000 por clase), incluida en `data/notmnist_sample_60k.npz`
- **División:** 80% entrenamiento (48,000) / 20% prueba (11,999), estratificada por clase
- **Modelo:** Regresión logística One-vs-All (10 clasificadores binarios), entrenados por descenso de gradiente con regularización L2
- **Resultado obtenido:** ~80.6% de precisión (accuracy) en el conjunto de prueba.

## Resumen de la metodología

1. **Preprocesamiento con Pandas**: indexación de rutas de imágenes por clase, muestreo estratificado balanceado (`groupby().sample()`), verificación de nulos y de balance de clases.
2. **Normalización**: valores de píxel escalados a [0, 1].
3. **División train/test**: partición 80/20 estratificada por clase, sin fuga de datos entre conjuntos (verificado explícitamente en el cuadernillo).
4. **Modelo**: regresión logística implementada desde cero (sigmoide, función de costo con regularización L2, descenso de gradiente vectorizado), aplicada bajo el esquema One-vs-All (10 clasificadores binarios).
5. **Evaluación**: curvas de costo por clasificador, curva de costo promedio, curva de precisión train/test vs. iteración, matriz de confusión, reporte de clasificación (precision/recall/F1) y ejemplos visuales de predicciones correctas e incorrectas.

## Resultados principales

| Métrica | Valor |
|---|---|
| Precisión (accuracy) — entrenamiento | 81.06% |
| Precisión (accuracy) — prueba | 80.63% |
| Iteraciones de entrenamiento | 400 |
| Tasa de aprendizaje (α) | 0.3 |
| Regularización (λ) | 1.0 |

No se observa sobreajuste relevante: las curvas de precisión de entrenamiento y prueba se mantienen muy cercanas durante todo el entrenamiento. Los errores de clasificación se concentran principalmente entre letras visualmente similares en ciertas tipografías, como se observa en la matriz de confusión.
