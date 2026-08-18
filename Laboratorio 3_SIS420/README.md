# Regresión Lineal Multivariable, Regresión Polinómica y Ecuación de la Normal — WEC Perth 49

Trabajo práctico modificado a partir de los cuadernillos revisados en clase `05_reg_pol_01.ipynb` y
`06_reg_pol_02.ipynb` (regresión polinómica).


## Dataset

**Large-scale Wave Energy Farm — Perth 49** (`data/WEC_Perth_49.csv`), UCI Machine Learning Repository.
Describe una granja de 49 convertidores de energía undimotriz (boyas) frente a la costa de Perth,
Australia. Cada fila es una configuración distinta del arreglo.

- **Entrada (X):** posiciones `X1,Y1 ... X10,Y10` de las primeras 10 boyas → **n = 20 características**
  (cumple n ≥ 10).
- **Objetivo (y):** `Total_Power`, potencia total generada por el arreglo (W).
- **m = 36 043 ejemplos** (cumple m ≥ 2 000).

## Contenido

```
.
├── data/
│   └── WEC_Perth_49.csv
├── notebooks/
│   └── 06_reg_pol_02_WEC.ipynb    # Cuadernillo principal (ejecutado, con salidas y graficas)
├── requirements.txt
└── README.md
```

## Qué contiene el cuadernillo

Los **tres modelos** se entrenan y validan dentro del mismo cuadernillo (`06_reg_pol_02_WEC.ipynb`):

1. **Regresión lineal multivariable** — descenso por el gradiente sobre las 20 características originales
   normalizadas. Incluye selección de α y gráfico de costo vs. iteración.
2. **Regresión polinómica (grado 2)** — características generadas con `PolynomialFeatures` (scikit-learn)
   a partir de las mismas 20 variables (230 características resultantes). Incluye selección de α y
   gráfico de costo vs. iteración.
3. **Ecuación de la normal** — solución cerrada aplicada tanto al modelo lineal como al polinómico
   (comparada gráficamente contra el resultado del descenso por el gradiente), y usada además para un
   análisis de costo vs. grado del polinomio (1 a 6) sobre dos variables, mostrando el comportamiento
   típico de sesgo/varianza.

Al final se presentan **150 predicciones** sobre el conjunto de prueba (nunca visto durante el
entrenamiento) del modelo polinómico, con un error porcentual promedio ≈ 1.06 %.

## Resultados principales

| Modelo | RMSE (prueba) | MAE (prueba) | R² (prueba) |
|---|---|---|---|
| Lineal multivariable | 83 974.82 W | 64 260.96 W | 0.5216 |
| Polinómico grado 2 | 64 973.57 W | 40 705.84 W | 0.7136 |

El modelo polinómico mejora claramente al lineal, evidenciando efectos de interacción no lineales entre
la posición de las boyas y la potencia total generada. En ambos casos, el costo del descenso por el
gradiente coincide prácticamente con el óptimo global de la ecuación de la normal.

## Cómo ejecutar

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/06_reg_pol_02_WEC.ipynb
```
