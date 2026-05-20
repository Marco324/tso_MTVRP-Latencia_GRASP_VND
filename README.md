# MTVRP-Latencia: Algoritmo GRASP para Ruteo de Vehículos con Múltiples Viajes y Mínima Latencia

Implementación en Python de un algoritmo heurístico GRASP con multiarranque para resolver el problema de ruteo de vehículos con un único vehículo, múltiples viajes desde el depósito y función objetivo de minimización de la suma de latencias de los clientes.

**Proyecto Integrador de Aprendizaje — Temas Selectos de Optimización.**

---

## Problema abordado

Un único vehículo realiza múltiples viajes desde un depósito para atender a todos los clientes. Cada viaje comienza y termina en el depósito, y cada cliente debe ser visitado exactamente una vez. Las restricciones son las siguientes:

- La suma de las demandas de los clientes asignados a un mismo viaje no puede exceder la capacidad del vehículo `Q`.
- Cuando se proporcione, la suma de las duraciones de todos los viajes no puede exceder el tiempo máximo de jornada `T`.

El objetivo **no es minimizar la distancia total recorrida** (objetivo clásico del VRP), sino la **suma de los tiempos de espera de los clientes** — su latencia total. La latencia de un cliente es el tiempo transcurrido desde que el vehículo sale por primera vez del depósito hasta el momento en que dicho cliente recibe el servicio. En presencia de múltiples viajes, los clientes de un viaje arrastran como tiempo de espera fijo la duración completa de todos los viajes previos.

---

## Método de solución

El algoritmo implementado es una metaheurística **GRASP (Greedy Randomized Adaptive Search Procedure) con multiarranque**, compuesta por cuatro elementos:

1. **Construcción golosa-aleatorizada.** Genera una solución factible viaje por viaje, eligiendo el siguiente cliente mediante una Lista Restringida de Candidatos (RCL) parametrizada por α.
2. **Búsqueda local VND.** Refina la solución mediante tres operadores de vecindario aplicados en orden: *relocate*, *swap* y *2-opt*, hasta alcanzar un óptimo local.
3. **Ordenamiento óptimo de viajes.** Regla exacta —no heurística— que ordena los viajes por la razón (duración del viaje / número de clientes). Se aplica tras la construcción y después de cada movimiento de la búsqueda local.
4. **Multiarranque.** Repite construcción + mejora durante un número predeterminado de iteraciones con distintas semillas aleatorias y conserva la mejor solución encontrada.

---

## Requisitos

- Python 3.8 o superior.
- Sin dependencias externas (solo módulos estándar: `math`, `random`, `time`, `argparse`, `os`).

---

## Estructura del proyecto

```
mtvrp_grasp.py        Código principal con el algoritmo
README.md             Este archivo
MT-DMP10s0-01.txt     Instancia oficial de ejemplo (formato matriz)
VRPNC1m.TXT           Instancia oficial de ejemplo (formato coordenadas)
```

---

## Ejecución

### 1. Validación inicial

Antes de la primera corrida conviene verificar que el cálculo de la función objetivo es correcto:

```
python3 mtvrp_grasp.py --selftest
```

Si imprime `Auto-prueba de latencia (con y sin servicio): OK`, el código está bien y se puede proceder.

### 2. Resolver una instancia

```
python3 mtvrp_grasp.py instancias/MT-DMP10s0-01.txt
```

### 3. Resolver varias instancias en una sola corrida

```
python3 mtvrp_grasp.py MT-DMP10s0-01.txt VRPNC1m.TXT
```

### Opciones disponibles

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--replicas N` | Número de réplicas independientes por configuración | 10 |
| `--time-limit S` | Límite de tiempo (segundos) por corrida individual | sin límite |
| `--round` | Redondea las distancias euclidianas a enteros | desactivado |
| `--selftest` | Solo ejecuta la auto-prueba de validación | — |

### Ejemplo de ejecución completa para el reporte final

```
python3 mtvrp_grasp.py MT-DMP10s0-01.txt VRPNC1m.TXT --replicas 10 --time-limit 80
```

---

## Formatos de instancia soportados

El lector **detecta el formato automáticamente** a partir del contenido del archivo.

### Formato 1: matriz explícita (`TravelTimes`)

Las distancias o tiempos de viaje entre cada par de nodos vienen dados directamente en una matriz de tamaño `(n+1) × (n+1)`, donde `n` es el número de clientes y el nodo 0 es el depósito.

```
nbClients: 10
nbTrips: 2
VehCapacity: 120
ClientDemands:
10  20  30  10  20  10  20  30  30  30
ServiceTimes:
0   0   0   0   0   0   0   0   0   0
TravelTimes:
0   33  54  14  70  92  79  80  61  58  14
33  0   21  18  42  58  49  55  28  27  36
...
```

### Formato 2: coordenadas (`CoorX/CoorY`)

Las distancias se calculan automáticamente mediante la fórmula euclidiana a partir de las coordenadas de cada nodo:

```
nbClients: 50
nbTrips: 5
VehCapacity 160
ClientDemands:
7  30  16  9  21 ...
ServiceTimes:
0  0  0  0  0 ...

CoorX   CoorY
30      40
37      52
49      49
...
```

La primera coordenada corresponde al depósito; las siguientes, a los clientes 1, 2, ..., n.

### Campos comunes

- `nbClients`: número de clientes (sin contar el depósito).
- `nbTrips`: número de viajes esperado (informativo).
- `VehCapacity`: capacidad del vehículo `Q`.
- `MaxTime` *(opcional)*: tiempo máximo de jornada `T`.

---

## Configuraciones del experimento

El experimento por defecto evalúa **cuatro configuraciones** que varían dos parámetros: el factor de aleatorización α de la RCL y el criterio de construcción goloso.

| Configuración | α | Criterio de construcción | Rol en el experimento |
|---|---|---|---|
| C1 | 0.2 | Vecino más cercano (`nn`) | Aleatorización baja |
| C2 | 0.4 | Vecino más cercano (`nn`) | Aleatorización alta |
| C3 | 0.3 | Inserción por incremento de llegada (`ins`) | Criterio constructivo alternativo |
| C4 | 0.0 | Vecino más cercano (`nn`) | Línea base (heurística golosa pura) |

La configuración C4 con α = 0 funciona como **grupo de control**: permite cuantificar el aporte específico de la fase de aleatorización al desempeño del algoritmo.

---

## Diseño experimental

- **Criterio de paro:** 80 iteraciones del bucle interno de GRASP por cada corrida (configurable mediante el código).
- **Réplicas:** 10 corridas independientes por configuración, cada una con una semilla aleatoria distinta. Valor estándar en la literatura de metaheurísticas para reportar mejor, promedio y peor desempeño.
- **Reproducibilidad:** las semillas son determinísticas (`base_seed = 1000` y se incrementa por réplica), por lo que dos ejecuciones del mismo comando producen exactamente los mismos resultados.

---

## Cómo interpretar la salida

Para cada instancia, el programa imprime primero una tabla comparativa de configuraciones y a continuación los detalles de la mejor solución encontrada:

```
===== Experimento: 'MT-DMP10s0-01'  (n=10, Q=120.0, T=None, nbTrips=2) =====

Config           Mejor    Promedio      Peor   T.prom(s)   Repl.
----------------------------------------------------------------------
C1-nn-a02      1401.00    1401.00    1401.00      0.100      10
C2-nn-a04      1401.00    1401.00    1401.00      0.144      10
C3-ins-a03     1401.00    1401.00    1401.00      0.131      10
C4-nn-a00      1401.00    1401.00    1401.00      0.066      10

Mejor configuracion: C1-nn-a02

--- Mejor solucion para la instancia 'MT-DMP10s0-01' ---
Funcion objetivo (suma de latencias): 1401.00
Numero de viajes usados: 2
  Viaje 1: 0 -> 3 -> 1 -> 2 -> 8 -> 9 -> 0
           carga=120/120  duracion=127.00
  Viaje 2: 0 -> 10 -> 4 -> 5 -> 6 -> 7 -> 0
           carga=90/120   duracion=248.00
  Verificacion (suma de latencias recalculada): 1401.00
```

- **Función objetivo**: suma total de latencias (es el valor que se minimiza).
- **Mejor / Promedio / Peor**: estadísticas calculadas sobre las réplicas de esa configuración.
- **T.prom(s)**: tiempo de cómputo promedio por réplica, en segundos.
- **Verificación**: la latencia recalculada de forma independiente debe coincidir con la función objetivo (es un control de correctitud).

---

## Notas sobre el modelo

- Los **tiempos de servicio** están incorporados al cálculo: el servicio en un cliente no cuenta dentro de su propia latencia (que se mide al momento de llegada), pero sí retrasa la llegada a los clientes posteriores.
- Las distancias euclidianas se mantienen como números reales por defecto (no se redondean), lo cual es lo más preciso para minimizar latencia. Para instancias clásicas tipo CVRP donde se acostumbra redondear, usar la opción `--round`.
- El número de viajes producido por el algoritmo suele coincidir con el mínimo dictado por la capacidad (`ceil(demanda total / Q)`). Como cada viaje adicional penaliza la latencia, el método naturalmente evita generar viajes innecesarios.

---

## Autores

*(Completar con los datos del equipo antes de la entrega)*

- **Nombre del equipo:**
- **Integrantes y matrículas:**
- **Materia:** Temas Selectos de Optimización
- **Institución:** Universidad Autónoma de Nuevo León (UANL)
- **Fecha de entrega:**