# Regresión Lineal Múltiple — Granja de convertidores de energía undimotriz (WEC Perth 49)

Trabajo práctico de **Regresión Lineal Múltiple**, modificado a partir del cuadernillo revisado en clases
`02_reg_lin_mul_01.ipynb` (basado en el ejercicio clásico de Andrew Ng de precios de casas de Portland).

En este trabajo se reemplaza el dataset original (47 ejemplos, 2 características) por un dataset real de
mayor escala que cumple los requisitos de la actividad:

- **n ≥ 20** características de entrada → **n = 98**
- **m ≥ 20 000** ejemplos → **m = 36 043**

## Dataset

**Large-scale Wave Energy Farm — Perth 49** (`data/WEC_Perth_49.csv`), publicado originalmente en el
UCI Machine Learning Repository (Neshat et al., *"Large-scale Wave Energy Farm"*).

El dataset describe una granja de **49 convertidores de energía undimotriz (WEC / boyas)** frente a la
costa de Perth (Australia). Cada fila representa una configuración distinta del arreglo:

| Columnas | Descripción |
|---|---|
| `X1..X49`, `Y1..Y49` | Coordenadas (x, y) en metros de cada una de las 49 boyas → **variables de entrada (X)** |
| `Power1..Power49` | Potencia individual generada por cada boya (no usada como entrada, se descarta para evitar fuga de información) |
| `qW` | Factor de calidad de interacción del arreglo (no usado como entrada) |
| `Total_Power` | Potencia total generada por el arreglo → **variable objetivo (y)** |

**Problema a resolver:** dado el layout geométrico (posición de las 49 boyas), inferir la **potencia
total** que generará el arreglo — un problema real de optimización de layout en energías renovables.

## Contenido del repositorio

```
.
├── data/
│   └── WEC_Perth_49.csv          # Dataset (m=36043, n=98 + variables auxiliares)
├── notebooks/
│   └── 02_reg_lin_mul_01_WEC.ipynb   # Cuadernillo principal (ejecutado, con salidas y graficas)
├── requirements.txt
└── README.md
```

## Metodología

1. **Carga y preprocesamiento con Pandas**: `pd.read_csv`, exploración (`head`, `info`, `describe`),
   verificación de valores nulos, selección de columnas de entrada/objetivo.
2. **División train/test** (80% / 20%) con semilla fija.
3. **Normalización de características** (z-score), calculada solo sobre el conjunto de entrenamiento
   y aplicada también al de prueba. También se normaliza la variable objetivo `Total_Power` (magnitud
   ≈ 4×10⁶ W) para estabilizar numéricamente el descenso por el gradiente.
4. **Función de costo** vectorizada `computeCostMulti` y **descenso por el gradiente**
   `gradientDescentMulti`, implementados en NumPy puro (misma formulación que el cuadernillo original).
5. **Selección del coeficiente de aprendizaje α**, comparando gráficamente la curva de costo vs.
   iteración para varios valores de α.
6. **Entrenamiento final** (α = 0.1, 1500 iteraciones) con gráfica de convergencia de J(θ) vs. número
   de iteración.
7. **Evaluación** sobre el conjunto de prueba: MSE, RMSE, MAE, R².
8. **Ecuación de la normal** como validación del óptimo global (problema convexo), comparada contra el
   resultado del descenso por el gradiente.
9. **Inferencia** de ejemplos nuevos (predicción de `Total_Power` a partir de un layout de boyas).

## Resultados

- Costo final (descenso por el gradiente, α=0.1, 1500 iter.): **J ≈ 0.0836** (variable objetivo
  normalizada), prácticamente idéntico al mínimo global calculado con la ecuación de la normal
  (diferencia relativa < 0.001%).
- Conjunto de prueba: **RMSE ≈ 49 732 W**, **R² ≈ 0.832**.

Todas las gráficas (comparación de α, convergencia del costo, predicción vs. valor real) se encuentran
generadas dentro del propio cuadernillo `notebooks/02_reg_lin_mul_01_WEC.ipynb`.

## Cómo ejecutar

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/02_reg_lin_mul_01_WEC.ipynb
```

## Autoría

Trabajo práctico de la asignatura de Machine Learning — Regresión Lineal Múltiple.
