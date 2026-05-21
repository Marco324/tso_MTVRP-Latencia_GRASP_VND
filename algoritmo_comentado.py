"""
===========================================================================
MTVRP (Multi-Trip Vehicle Routing Problem) con MINIMA LATENCIA
Algoritmo: GRASP + Busqueda Local VND + Multiarranque
===========================================================================

DESCRIPCION DEL PROBLEMA:
    Un unico vehiculo realiza MULTIPLES VIAJES desde un deposito para
    atender a TODOS los clientes. Cada viaje empieza y termina en el
    deposito. Hay dos restricciones:
        1. Capacidad maxima Q del vehiculo (por viaje).
        2. Tiempo maximo T de jornada (suma de todos los viajes), opcional.

OBJETIVO:
    Minimizar la SUMA DE LATENCIAS de los clientes. La latencia de un
    cliente es el tiempo que espera desde que el vehiculo sale por
    primera vez del deposito hasta que ese cliente recibe servicio.

IDEA CLAVE:
    Para el viaje numero k, todos sus clientes "arrastran" como espera
    fija la suma de las duraciones COMPLETAS de los viajes anteriores
    (1, 2, ..., k-1). Eso se llama OFFSET. Dentro del viaje, la latencia
    de cada cliente es su tiempo de llegada acumulado.

        Latencia_total = suma_sobre_k [ n_k * offset_k + latencia_interna_k ]

    donde:
        n_k        = numero de clientes del viaje k
        offset_k   = suma de duraciones de los viajes 1, 2, ..., k-1
        latencia_interna_k = suma de tiempos de llegada dentro del viaje k

    NOTA: el tiempo de servicio de un cliente NO cuenta en su propia
    latencia (se mide AL LLEGAR), pero SI retrasa a los clientes
    que vienen despues en el mismo viaje.

REGLA TEORICA APROVECHADA:
    Dado un conjunto FIJO de viajes, el orden optimo entre ellos se
    obtiene ordenandolos de forma CRECIENTE por el ratio:
            duracion_del_viaje / numero_de_clientes_del_viaje
    Se demuestra por un argumento de intercambio entre viajes adyacentes.

FORMATOS DE INSTANCIA SOPORTADOS:
    (1) Oficial con MATRIZ explicita (TravelTimes) - p.ej. MT-DMP10s0-01
    (2) Oficial con COORDENADAS (CoorX/CoorY)      - p.ej. VRPNC1m
    (3) Formato simple (NODES) y CVRP/TSPLIB       - por compatibilidad

Autor: (completar con datos del equipo)
===========================================================================
"""

import math       # para sqrt, ceil, hypot (distancia euclidiana)
import random     # para el generador aleatorio reproducible
import time       # para medir tiempos de ejecucion
import argparse   # para procesar argumentos de linea de comandos
import os         # para manejo de rutas y archivos


# ===========================================================================
# 1. REPRESENTACION DE LA INSTANCIA
# ===========================================================================

class Instancia:
    """
    Contenedor con todos los datos de UNA instancia del MTVRP.

    Convencion importante:
        El indice 0 es SIEMPRE el deposito.
        Los clientes son los indices 1, 2, ..., numero_clientes.
    """

    def __init__(self, nombre, demandas, capacidad_vehiculo,
                 matriz_distancias, tiempos_servicio=None,
                 tiempo_max_jornada=None, num_viajes_sugerido=None):
        """
        ENTRADAS:
            nombre               : str, identificador de la instancia.
            demandas             : lista, demandas[0]=0 (deposito), 
                                   demandas[i]=demanda del cliente i.
            capacidad_vehiculo   : float, capacidad maxima Q por viaje.
            matriz_distancias    : matriz cuadrada (n+1)x(n+1) con
                                   tiempos/distancias entre todos los nodos.
            tiempos_servicio     : lista opcional, tiempo de atencion en
                                   cada nodo (0 si no se especifica).
            tiempo_max_jornada   : float opcional, tiempo total maximo T.
            num_viajes_sugerido  : int opcional, dato informativo de la
                                   instancia (no es restriccion estricta).

        SALIDA (lo que queda guardado en el objeto):
            Todos los datos anteriores, mas:
                numero_clientes  : n
                deposito         : 0 (constante)
                demanda_total    : suma de todas las demandas
                viajes_minimos   : cota inferior teorica de viajes
                                   = techo(demanda_total / Q)
        """
        self.nombre = nombre
        self.demandas = demandas
        self.numero_clientes = len(demandas) - 1   # se resta el deposito
        self.deposito = 0
        self.capacidad_vehiculo = capacidad_vehiculo
        self.tiempo_max_jornada = tiempo_max_jornada
        self.num_viajes_sugerido = num_viajes_sugerido
        self.matriz_distancias = matriz_distancias

        # Si no nos dan tiempos de servicio, se asume 0 para todos.
        if tiempos_servicio is not None:
            self.tiempos_servicio = tiempos_servicio
        else:
            self.tiempos_servicio = [0.0] * len(demandas)

        self.demanda_total = sum(demandas)

        # Cota inferior: aunque hicieramos viajes perfectamente llenos,
        # necesitariamos al menos techo(demanda_total / Q) viajes.
        if capacidad_vehiculo:
            self.viajes_minimos = math.ceil(
                self.demanda_total / capacidad_vehiculo)
        else:
            self.viajes_minimos = None

    def __repr__(self):
        """Representacion en texto del objeto, util para depurar."""
        return (f"Instancia(nombre={self.nombre}, "
                f"n={self.numero_clientes}, Q={self.capacidad_vehiculo}, "
                f"T={self.tiempo_max_jornada}, "
                f"nbTrips={self.num_viajes_sugerido}, "
                f"demanda_total={self.demanda_total:.0f})")


# ===========================================================================
# 2. LECTURA DE INSTANCIAS DESDE ARCHIVO
# ===========================================================================

def construir_matriz_euclidiana(lista_coordenadas, redondear=False):
    """
    Calcula la matriz de distancias euclidianas a partir de coordenadas.

    ENTRADA:
        lista_coordenadas : lista de tuplas (x, y), una por nodo.
                            El primer elemento es el deposito.
        redondear         : bool, si True redondea cada distancia a
                            entero (algunas instancias TSPLIB lo piden).

    SALIDA:
        matriz cuadrada (lista de listas) con la distancia euclidiana
        entre cada par de nodos.
    """
    numero_nodos = len(lista_coordenadas)
    matriz = [[0.0] * numero_nodos for _ in range(numero_nodos)]
    for i in range(numero_nodos):
        x_i, y_i = lista_coordenadas[i]
        for j in range(numero_nodos):
            x_j, y_j = lista_coordenadas[j]
            distancia = math.hypot(x_i - x_j, y_i - y_j)
            if redondear:
                matriz[i][j] = float(round(distancia))
            else:
                matriz[i][j] = distancia
    return matriz


def extraer_ultimo_numero(linea):
    """
    Devuelve el ultimo token numerico de una linea tipo 'clave: valor'.

    ENTRADA:
        linea : str, por ejemplo 'nbClients: 10' o 'VehCapacity 200'.

    SALIDA:
        str con el ultimo token (que se debe convertir a float despues).
    """
    return linea.replace(":", " ").split()[-1]


def leer_formato_oficial(lineas, nombre, redondear=False):
    """
    Lee instancias en el formato OFICIAL del proyecto.

    Claves reconocidas (sin importar si vienen con ':' o no):
        nbClients      -> numero de clientes
        nbTrips        -> numero de viajes sugerido (informativo)
        VehCapacity    -> capacidad del vehiculo
        MaxTime        -> tiempo maximo de jornada (opcional)
        ClientDemands  -> n valores con las demandas
        ServiceTimes   -> n valores con tiempos de servicio
        TravelTimes    -> matriz (n+1)x(n+1), nodo 0 = deposito
        CoorX / CoorY  -> (n+1) pares de coordenadas

    Los numeros pueden venir todos en una linea o repartidos en varias.

    ENTRADA:
        lineas    : lista de strings, ya limpias (sin lineas vacias).
        nombre    : str, nombre que se le pondra a la Instancia.
        redondear : bool, redondear distancias euclidianas si aplica.

    SALIDA:
        objeto Instancia listo para usar.
    """
    # Variables que iremos llenando al recorrer el archivo:
    numero_clientes = None
    num_viajes_sugerido = None
    capacidad_vehiculo = None
    tiempo_max_jornada = None

    # Listas temporales para acumular los numeros de cada seccion:
    demandas_brutas = []        # demandas leidas
    servicios_brutos = []       # tiempos de servicio leidos
    matriz_bruta = []           # numeros sueltos de la matriz
    coordenadas_brutas = []     # numeros sueltos de coordenadas

    # Que seccion estamos leyendo en este momento.
    seccion_actual = None

    for linea in lineas:
        linea_minusculas = linea.lower()

        # ----- Deteccion de encabezados (cambian la seccion actual) -----
        if linea_minusculas.startswith("nbclients"):
            numero_clientes = int(float(extraer_ultimo_numero(linea)))
            seccion_actual = None
        elif linea_minusculas.startswith("nbtrips"):
            num_viajes_sugerido = int(float(extraer_ultimo_numero(linea)))
            seccion_actual = None
        elif linea_minusculas.startswith("vehcapacity"):
            capacidad_vehiculo = float(extraer_ultimo_numero(linea))
            seccion_actual = None
        elif linea_minusculas.startswith(
                ("maxtime", "max_time", "maxduration", "jornada")):
            tiempo_max_jornada = float(extraer_ultimo_numero(linea))
            seccion_actual = None
        elif linea_minusculas.startswith("clientdemands"):
            seccion_actual = "demandas"
        elif linea_minusculas.startswith("servicetimes"):
            seccion_actual = "servicios"
        elif linea_minusculas.startswith("traveltimes"):
            seccion_actual = "matriz"
        elif (linea_minusculas.startswith("coorx")
              or ("coor" in linea_minusculas
                  and "coory" in linea_minusculas)):
            seccion_actual = "coordenadas"
        else:
            # ----- Linea con numeros sueltos: pertenece a alguna seccion --
            tokens = linea.replace(",", " ").split()
            if not tokens:
                continue
            try:
                numeros = [float(t) for t in tokens]
            except ValueError:
                # Si la linea no tiene solo numeros, la ignoramos.
                continue
            if seccion_actual == "demandas":
                demandas_brutas += numeros
            elif seccion_actual == "servicios":
                servicios_brutos += numeros
            elif seccion_actual == "matriz":
                matriz_bruta += numeros
            elif seccion_actual == "coordenadas":
                coordenadas_brutas += numeros

    # Validacion minima.
    if numero_clientes is None or capacidad_vehiculo is None:
        raise ValueError(
            "Faltan nbClients o VehCapacity en la instancia.")

    # Armar el vector de demandas con el deposito al inicio (valor 0).
    demandas = [0.0] + demandas_brutas[:numero_clientes]

    # Armar los tiempos de servicio (0 para el deposito).
    if servicios_brutos:
        tiempos_servicio = [0.0] + servicios_brutos[:numero_clientes]
    else:
        tiempos_servicio = [0.0] * (numero_clientes + 1)

    # Construir la matriz de distancias: o ya viene explicita, o se
    # calcula a partir de las coordenadas.
    if matriz_bruta:
        tamano = numero_clientes + 1
        if len(matriz_bruta) < tamano * tamano:
            raise ValueError("La matriz TravelTimes esta incompleta.")
        matriz_distancias = [
            matriz_bruta[i * tamano:(i + 1) * tamano]
            for i in range(tamano)
        ]
    elif coordenadas_brutas:
        puntos = [
            (coordenadas_brutas[2 * i], coordenadas_brutas[2 * i + 1])
            for i in range(numero_clientes + 1)
        ]
        matriz_distancias = construir_matriz_euclidiana(puntos, redondear)
    else:
        raise ValueError(
            "No se encontro ni TravelTimes ni coordenadas en el archivo.")

    return Instancia(
        nombre=nombre,
        demandas=demandas,
        capacidad_vehiculo=capacidad_vehiculo,
        matriz_distancias=matriz_distancias,
        tiempos_servicio=tiempos_servicio,
        tiempo_max_jornada=tiempo_max_jornada,
        num_viajes_sugerido=num_viajes_sugerido,
    )


def leer_instancia(ruta_archivo, redondear=False):
    """
    Lector general con DETECCION AUTOMATICA del formato.

    Reglas de deteccion:
      - Si aparece 'nbClients'        -> formato oficial del proyecto.
      - Si aparece 'NODE_COORD_SECTION' o 'DEMAND_SECTION' -> TSPLIB/CVRP.
      - En otro caso                  -> formato simple con seccion NODES.

    ENTRADA:
        ruta_archivo : str, ruta al archivo de la instancia.
        redondear    : bool, redondear distancias euclidianas si aplica.

    SALIDA:
        objeto Instancia listo para usar.
    """
    with open(ruta_archivo, "r") as archivo:
        contenido_crudo = archivo.read().replace("\r", "")

    # Lineas limpias (sin saltos finales) y filtradas (sin vacias).
    todas_las_lineas = [ln.strip() for ln in contenido_crudo.split("\n")]
    lineas_no_vacias = [ln for ln in todas_las_lineas if ln != ""]

    # Nombre por defecto: el del archivo sin extension.
    nombre = os.path.splitext(os.path.basename(ruta_archivo))[0]

    # Cadena auxiliar para detectar palabras clave.
    texto_completo = " ".join(ln.lower() for ln in lineas_no_vacias)

    # ------- CASO 1: formato oficial del proyecto -------
    if "nbclients" in texto_completo:
        return leer_formato_oficial(lineas_no_vacias, nombre, redondear)

    # ------- CASO 2: formato TSPLIB / CVRP -------
    if ("node_coord_section" in texto_completo
            or "demand_section" in texto_completo):
        capacidad = None
        tiempo_maximo = None
        coordenadas_por_id = {}
        demandas_por_id = {}
        id_deposito = None
        seccion_actual = None

        for linea in lineas_no_vacias:
            mayus = linea.upper()
            if mayus.startswith("NAME"):
                nombre = linea.split()[-1]
            elif mayus.startswith("CAPACITY"):
                capacidad = float(linea.replace(":", " ").split()[-1])
            elif mayus.startswith(("MAX_TIME", "DISTANCE", "MAX_DURATION")):
                tiempo_maximo = float(linea.replace(":", " ").split()[-1])
            elif mayus.startswith("NODE_COORD_SECTION"):
                seccion_actual = "coordenadas"
            elif mayus.startswith("DEMAND_SECTION"):
                seccion_actual = "demandas"
            elif mayus.startswith("DEPOT_SECTION"):
                seccion_actual = "deposito"
            elif mayus.startswith(("EOF", "DISPLAY_DATA")):
                seccion_actual = None
            elif mayus.startswith(("TYPE", "COMMENT", "DIMENSION",
                                    "EDGE_WEIGHT")):
                continue
            else:
                tokens = linea.split()
                if seccion_actual == "coordenadas" and len(tokens) >= 3:
                    coordenadas_por_id[int(tokens[0])] = (
                        float(tokens[1]), float(tokens[2]))
                elif seccion_actual == "demandas" and len(tokens) >= 2:
                    demandas_por_id[int(tokens[0])] = float(tokens[1])
                elif seccion_actual == "deposito":
                    valor = int(tokens[0])
                    if valor != -1 and id_deposito is None:
                        id_deposito = valor

        # Si no marcaron el deposito, asumimos que es el de menor ID.
        if id_deposito is None:
            id_deposito = min(coordenadas_por_id)

        # Reordenar: el deposito primero, los clientes despues.
        ids_clientes = sorted(
            k for k in coordenadas_por_id if k != id_deposito)
        orden = [id_deposito] + ids_clientes

        puntos = [coordenadas_por_id[i] for i in orden]
        demandas = [demandas_por_id.get(i, 0.0) for i in orden]
        demandas[0] = 0.0   # el deposito no tiene demanda

        return Instancia(
            nombre=nombre,
            demandas=demandas,
            capacidad_vehiculo=capacidad,
            matriz_distancias=construir_matriz_euclidiana(puntos, redondear),
            tiempo_max_jornada=tiempo_maximo,
        )

    # ------- CASO 3: formato simple con seccion NODES -------
    capacidad = None
    tiempo_maximo = None
    coordenadas_por_id = {}
    demandas_por_id = {}
    seccion_actual = None

    for linea in lineas_no_vacias:
        mayus = linea.upper()
        if mayus.startswith("NAME"):
            nombre = linea.split()[-1]
        elif mayus.startswith("CAPACITY"):
            capacidad = float(linea.split()[-1])
        elif mayus.startswith(("MAX_TIME", "MAX_DURATION", "JORNADA")):
            tiempo_maximo = float(linea.split()[-1])
        elif mayus.startswith(("DIMENSION", "COMMENT", "TYPE")):
            continue
        elif mayus.startswith(("NODES", "NODE_SECTION", "DATA")):
            seccion_actual = "nodos"
        else:
            tokens = linea.split()
            if seccion_actual == "nodos" and len(tokens) >= 4:
                indice = int(tokens[0])
                coordenadas_por_id[indice] = (float(tokens[1]),
                                              float(tokens[2]))
                demandas_por_id[indice] = float(tokens[3])

    # Deposito: el de menor ID o el que tenga demanda 0.
    id_deposito = min(coordenadas_por_id)
    for clave, valor in demandas_por_id.items():
        if valor == 0:
            id_deposito = clave
            break

    ids_clientes = sorted(
        k for k in coordenadas_por_id if k != id_deposito)
    orden = [id_deposito] + ids_clientes
    puntos = [coordenadas_por_id[i] for i in orden]
    demandas = [demandas_por_id.get(i, 0.0) for i in orden]
    demandas[0] = 0.0

    return Instancia(
        nombre=nombre,
        demandas=demandas,
        capacidad_vehiculo=capacidad,
        matriz_distancias=construir_matriz_euclidiana(puntos, redondear),
        tiempo_max_jornada=tiempo_maximo,
    )


# ===========================================================================
# 3. EVALUACION DE LA SOLUCION (FUNCION OBJETIVO DE LATENCIA)
# ===========================================================================
#
# Representacion de una solucion:
#   solucion = [viaje_1, viaje_2, ..., viaje_m]
#   cada viaje es una lista ordenada de clientes, ej: [3, 7, 2]
#   El deposito (0) es IMPLICITO al inicio y al final de cada viaje.
#
# ===========================================================================

def calcular_duracion_viaje(viaje, instancia):
    """
    Calcula la duracion COMPLETA de un viaje:
        deposito -> cliente_1 -> cliente_2 -> ... -> deposito.
    Incluye los tiempos de servicio en cada cliente.

    ENTRADA:
        viaje     : lista de clientes (sin el deposito).
        instancia : objeto Instancia.

    SALIDA:
        float con la duracion total del viaje (tiempo o distancia).
    """
    if not viaje:
        return 0.0

    distancias = instancia.matriz_distancias
    servicios = instancia.tiempos_servicio

    tiempo_acumulado = 0.0
    nodo_previo = 0   # arrancamos en el deposito

    # Recorremos el viaje cliente por cliente.
    for cliente_actual in viaje:
        tiempo_acumulado += distancias[nodo_previo][cliente_actual]
        tiempo_acumulado += servicios[cliente_actual]
        nodo_previo = cliente_actual

    # Falta el regreso del ultimo cliente al deposito.
    tiempo_acumulado += distancias[nodo_previo][0]

    return tiempo_acumulado


def calcular_latencia_interna_viaje(viaje, instancia):
    """
    Suma de los TIEMPOS DE LLEGADA de los clientes de UN viaje, asumiendo
    que el viaje arranca en tiempo 0 (es decir, sin contar el offset que
    vendria de viajes anteriores).

    Recordatorio importante: el servicio de un cliente NO cuenta en su
    propia latencia, pero SI retrasa a los siguientes en el mismo viaje.

    ENTRADA:
        viaje     : lista de clientes.
        instancia : objeto Instancia.

    SALIDA:
        float con la suma de latencias internas.
    """
    if not viaje:
        return 0.0

    distancias = instancia.matriz_distancias
    servicios = instancia.tiempos_servicio

    tiempo_llegada_actual = 0.0   # cuando llegamos al cliente actual
    suma_latencias_internas = 0.0
    nodo_previo = 0

    for cliente_actual in viaje:
        # El tiempo de llegada al cliente es el acumulado + el arco.
        tiempo_llegada_actual += distancias[nodo_previo][cliente_actual]
        # Esa llegada es su latencia interna en este viaje.
        suma_latencias_internas += tiempo_llegada_actual
        # Despues de atenderlo, sumamos el servicio (afecta a los siguientes).
        tiempo_llegada_actual += servicios[cliente_actual]
        nodo_previo = cliente_actual

    return suma_latencias_internas


def evaluar_solucion(solucion, instancia, penalizacion=1e9):
    """
    Calcula el VALOR OBJETIVO (suma total de latencias) de una solucion
    completa, y revisa si es factible (capacidad y tiempo de jornada).

    Si la solucion es INFACTIBLE, se le aplica una gran penalizacion para
    que la busqueda local la rechace automaticamente.

    ENTRADA:
        solucion     : lista de viajes [[...], [...], ...].
        instancia    : objeto Instancia.
        penalizacion : float, valor base para penalizar infactibilidad.

    SALIDA:
        tupla (valor_objetivo, es_factible)
        - valor_objetivo : float, suma de latencias (+ penalizacion si aplica).
        - es_factible    : bool, True si cumple capacidad y jornada.
    """
    # Filtramos viajes vacios (defensivo).
    viajes_no_vacios = [v for v in solucion if v]

    suma_latencias_total = 0.0
    offset_acumulado = 0.0    # offset_k de la formula
    es_factible = True
    exceso_total = 0.0        # cuanto nos pasamos en lo infactible
    tiempo_jornada_total = 0.0

    for viaje in viajes_no_vacios:
        # ----- Restriccion 1: capacidad del vehiculo -----
        carga_del_viaje = sum(
            instancia.demandas[cliente] for cliente in viaje)
        if carga_del_viaje > instancia.capacidad_vehiculo + 1e-9:
            es_factible = False
            exceso_total += (carga_del_viaje
                             - instancia.capacidad_vehiculo)

        # ----- Contribucion del viaje k a la latencia total -----
        numero_clientes_k = len(viaje)
        contribucion_viaje = (
            numero_clientes_k * offset_acumulado
            + calcular_latencia_interna_viaje(viaje, instancia)
        )
        suma_latencias_total += contribucion_viaje

        # ----- Actualizar offset para el siguiente viaje -----
        duracion_k = calcular_duracion_viaje(viaje, instancia)
        offset_acumulado += duracion_k
        tiempo_jornada_total += duracion_k

    # ----- Restriccion 2: tiempo maximo de jornada T (si existe) -----
    if instancia.tiempo_max_jornada is not None:
        if tiempo_jornada_total > instancia.tiempo_max_jornada + 1e-6:
            es_factible = False
            exceso_total += (tiempo_jornada_total
                             - instancia.tiempo_max_jornada)

    # Si hay infactibilidad, penalizamos de forma que la busqueda local
    # nunca acepte una solucion infactible como mejora.
    if not es_factible:
        suma_latencias_total += penalizacion + exceso_total * penalizacion

    return suma_latencias_total, es_factible


def calcular_latencia_por_cliente(solucion, instancia):
    """
    Calcula la latencia INDIVIDUAL de cada cliente.
    Util para VERIFICAR que la formula condensada coincide con la suma
    de latencias individuales (auto-verificacion).

    ENTRADA:
        solucion  : lista de viajes.
        instancia : objeto Instancia.

    SALIDA:
        diccionario {cliente: latencia_de_ese_cliente}.
    """
    distancias = instancia.matriz_distancias
    servicios = instancia.tiempos_servicio
    latencia_por_cliente = {}
    offset_acumulado = 0.0

    for viaje in solucion:
        if not viaje:
            continue
        tiempo_dentro_del_viaje = 0.0
        nodo_previo = 0
        for cliente_actual in viaje:
            tiempo_dentro_del_viaje += distancias[nodo_previo][cliente_actual]
            # Latencia = offset (lo arrastrado) + llegada dentro del viaje.
            latencia_por_cliente[cliente_actual] = (
                offset_acumulado + tiempo_dentro_del_viaje)
            tiempo_dentro_del_viaje += servicios[cliente_actual]
            nodo_previo = cliente_actual
        # Cerrar el viaje sumando el regreso al deposito para el offset.
        offset_acumulado += (tiempo_dentro_del_viaje
                             + distancias[nodo_previo][0])

    return latencia_por_cliente


# ===========================================================================
# 4. ORDEN OPTIMO ENTRE VIAJES (REGLA DE INTERCAMBIO)
# ===========================================================================

def ordenar_viajes_de_forma_optima(solucion, instancia):
    """
    Reordena los viajes para MINIMIZAR la latencia total, sin cambiar el
    contenido de cada viaje.

    REGLA EXACTA:
        Los viajes se ordenan ASCENDENTEMENTE por el ratio:
                duracion_del_viaje / numero_de_clientes_del_viaje
        Esto se demuestra con un argumento de intercambio entre dos
        viajes adyacentes en la secuencia.

    INTUICION:
        Conviene poner primero los viajes que aportan poco offset (son
        cortos) pero benefician a muchos clientes (los del propio viaje
        y los de los viajes que vienen despues).

    ENTRADA:
        solucion  : lista de viajes (algunos pueden estar vacios).
        instancia : objeto Instancia.

    SALIDA:
        nueva lista de viajes, sin vacios, en el orden optimo.
    """
    viajes_no_vacios = [v for v in solucion if v]

    def ratio_del_viaje(viaje):
        numero_clientes_k = len(viaje)
        if numero_clientes_k == 0:
            return float("inf")
        duracion = calcular_duracion_viaje(viaje, instancia)
        return duracion / numero_clientes_k

    viajes_no_vacios.sort(key=ratio_del_viaje)
    return viajes_no_vacios


# ===========================================================================
# 5. CONSTRUCCION GOLOSA ALEATORIZADA (FASE DE CONSTRUCCION DE GRASP)
# ===========================================================================

def construir_solucion_grasp(instancia, alpha, generador_aleatorio,
                              modo="vecino_mas_cercano"):
    """
    Construye una solucion INICIAL usando un metodo goloso aleatorizado,
    al estilo GRASP. Se arman viajes uno por uno, agregando clientes
    seleccionados de una Restricted Candidate List (RCL).

    PARAMETRO ALPHA:
        alpha = 0.0 -> goloso puro: siempre elige al mejor candidato.
        alpha = 1.0 -> totalmente aleatorio: todos los candidatos pueden
                       entrar en la RCL.
        Valores intermedios dan distintos grados de aleatoriedad.

    COMO SE ARMA LA RCL (para cada paso):
        - Se calcula la "puntuacion" de cada candidato (aqui: distancia
          desde el nodo actual).
        - Sea g_min la mejor puntuacion y g_max la peor.
        - Umbral = g_min + alpha * (g_max - g_min).
        - RCL = candidatos con puntuacion <= umbral.
        - Se elige UNO al azar de la RCL.

    ENTRADA:
        instancia            : objeto Instancia.
        alpha                : float entre 0 y 1, controla la aleatoriedad.
        generador_aleatorio  : objeto random.Random (para reproducibilidad).
        modo                 : str, en este codigo se mantiene por
                               compatibilidad; "vecino_mas_cercano" usa
                               la distancia desde el nodo actual.

    SALIDA:
        lista de viajes, ya reordenada con la regla optima entre viajes.
    """
    distancias = instancia.matriz_distancias

    # Conjunto de clientes que aun no han sido visitados.
    clientes_sin_visitar = set(range(1, instancia.numero_clientes + 1))

    solucion_en_construccion = []

    # Mientras queden clientes, abrimos un nuevo viaje.
    while clientes_sin_visitar:
        viaje_actual = []
        capacidad_restante = instancia.capacidad_vehiculo
        nodo_actual = 0   # cada viaje arranca en el deposito

        while True:
            # Candidatos: clientes que aun caben en el camion.
            candidatos = [
                cliente for cliente in clientes_sin_visitar
                if instancia.demandas[cliente] <= capacidad_restante + 1e-9
            ]
            if not candidatos:
                # Ya no cabe ningun cliente mas en este viaje: lo cerramos.
                break

            # Calcular puntuacion (distancia desde nodo_actual) y ordenar.
            candidatos_puntuados = sorted(
                (distancias[nodo_actual][cliente], cliente)
                for cliente in candidatos
            )
            puntuacion_minima = candidatos_puntuados[0][0]
            puntuacion_maxima = candidatos_puntuados[-1][0]

            # Umbral de la RCL.
            umbral = (puntuacion_minima
                      + alpha * (puntuacion_maxima - puntuacion_minima))

            # Construir la Restricted Candidate List.
            lista_rcl = [
                cliente
                for (puntuacion, cliente) in candidatos_puntuados
                if puntuacion <= umbral + 1e-9
            ]

            # Elegir uno al azar de la RCL.
            cliente_elegido = generador_aleatorio.choice(lista_rcl)

            # Agregarlo al viaje y actualizar el estado.
            viaje_actual.append(cliente_elegido)
            clientes_sin_visitar.discard(cliente_elegido)
            capacidad_restante -= instancia.demandas[cliente_elegido]
            nodo_actual = cliente_elegido

        if not viaje_actual:
            # Esto solo pasa si algun cliente tiene demanda > Q.
            raise ValueError(
                "Existe un cliente con demanda mayor que la capacidad Q.")

        solucion_en_construccion.append(viaje_actual)

    # Antes de devolver, reordenamos los viajes con la regla optima.
    return ordenar_viajes_de_forma_optima(solucion_en_construccion,
                                          instancia)


# ===========================================================================
# 6. BUSQUEDA LOCAL VND (relocate, swap, 2-opt) + REORDEN DE VIAJES
# ===========================================================================

def capacidad_respetada(viaje, instancia):
    """
    Verifica si la suma de demandas de un viaje no excede la capacidad Q.

    ENTRADA:
        viaje     : lista de clientes.
        instancia : objeto Instancia.

    SALIDA:
        bool: True si carga <= Q, False si la rebasa.
    """
    carga_total = sum(instancia.demandas[c] for c in viaje)
    return carga_total <= instancia.capacidad_vehiculo + 1e-9


def busqueda_local_vnd(solucion_inicial, instancia):
    """
    Aplica BUSQUEDA LOCAL tipo VND (Variable Neighborhood Descent) con
    tres vecindarios, en este orden:
        1. RELOCATE   : mover un cliente a otra posicion (mismo o otro viaje).
        2. SWAP       : intercambiar dos clientes entre si.
        3. 2-OPT      : invertir un segmento dentro de un mismo viaje.

    ESTRATEGIA:
        - First-improvement (primera mejora): apenas se encuentra un
          movimiento que mejora, se aplica y se reinicia desde el
          vecindario 1.
        - Esto se repite hasta que NINGUN vecindario logra mejorar
          (optimo local respecto a VND).

    DETALLE IMPORTANTE PARA LATENCIA:
        En cada movimiento candidato se aplica ordenar_viajes_de_forma_optima
        ANTES de evaluar. Esto es esencial porque la funcion objetivo es
        muy sensible al orden de los viajes (por los offsets acumulados).

    ENTRADA:
        solucion_inicial : lista de viajes (cualquier solucion factible).
        instancia        : objeto Instancia.

    SALIDA:
        tupla (solucion_optima_local, valor_objetivo)
    """
    # Copia profunda de los viajes para no modificar el original.
    solucion_actual = [list(viaje) for viaje in solucion_inicial if viaje]
    mejor_valor, _ = evaluar_solucion(solucion_actual, instancia)

    hubo_mejora = True
    while hubo_mejora:
        hubo_mejora = False

        # ============================================================
        # VECINDARIO 1: RELOCATE
        # Idea: por cada cliente, probar todas las posiciones posibles
        # (en su propio viaje o en cualquier otro viaje).
        # ============================================================
        for indice_viaje_origen in range(len(solucion_actual)):
            for posicion_origen in range(
                    len(solucion_actual[indice_viaje_origen])):

                cliente_a_mover = (
                    solucion_actual[indice_viaje_origen][posicion_origen])

                for indice_viaje_destino in range(len(solucion_actual)):
                    # Las posiciones de insercion van de 0 a len(viaje).
                    for posicion_destino in range(
                            len(solucion_actual[indice_viaje_destino]) + 1):

                        # Evitar mover a la misma posicion (sin efecto).
                        if (indice_viaje_origen == indice_viaje_destino
                                and (posicion_destino == posicion_origen
                                     or posicion_destino
                                        == posicion_origen + 1)):
                            continue

                        # Construir solucion candidata: clonar y mover.
                        solucion_candidata = [
                            list(v) for v in solucion_actual]

                        # Quitar el cliente de su posicion original.
                        solucion_candidata[
                            indice_viaje_origen].pop(posicion_origen)

                        # Ajustar la posicion de insercion si estamos en el
                        # mismo viaje y movemos hacia adelante.
                        if (indice_viaje_origen == indice_viaje_destino
                                and posicion_destino > posicion_origen):
                            solucion_candidata[
                                indice_viaje_destino].insert(
                                    posicion_destino - 1, cliente_a_mover)
                        else:
                            solucion_candidata[
                                indice_viaje_destino].insert(
                                    posicion_destino, cliente_a_mover)

                        # Eliminar viajes que hayan quedado vacios.
                        solucion_candidata = [
                            v for v in solucion_candidata if v]

                        # Verificar capacidad en TODOS los viajes.
                        if not all(capacidad_respetada(v, instancia)
                                   for v in solucion_candidata):
                            continue

                        # Reordenar viajes (clave para latencia minima).
                        solucion_candidata = ordenar_viajes_de_forma_optima(
                            solucion_candidata, instancia)

                        # Evaluar y comparar.
                        valor_candidato, factible = evaluar_solucion(
                            solucion_candidata, instancia)
                        if (factible
                                and valor_candidato < mejor_valor - 1e-7):
                            # Aceptamos el movimiento (first-improvement).
                            solucion_actual = [
                                list(v) for v in solucion_candidata]
                            mejor_valor = valor_candidato
                            hubo_mejora = True
                            break   # salir de bucle posicion_destino
                    if hubo_mejora:
                        break       # salir de bucle indice_viaje_destino
                if hubo_mejora:
                    break           # salir de bucle posicion_origen
            if hubo_mejora:
                break               # salir de bucle indice_viaje_origen

        if hubo_mejora:
            # Volver al inicio del while para reintentar desde RELOCATE.
            continue

        # ============================================================
        # VECINDARIO 2: SWAP
        # Idea: intercambiar dos clientes en cualesquiera posiciones.
        # ============================================================
        # Lista plana de todas las posiciones de clientes en la solucion.
        lista_posiciones = [
            (indice_viaje, posicion)
            for indice_viaje in range(len(solucion_actual))
            for posicion in range(len(solucion_actual[indice_viaje]))
        ]

        for a in range(len(lista_posiciones)):
            for b in range(a + 1, len(lista_posiciones)):
                viaje_a, pos_a = lista_posiciones[a]
                viaje_b, pos_b = lista_posiciones[b]

                # Construir solucion candidata con el intercambio.
                solucion_candidata = [list(v) for v in solucion_actual]
                (solucion_candidata[viaje_a][pos_a],
                 solucion_candidata[viaje_b][pos_b]) = (
                    solucion_candidata[viaje_b][pos_b],
                    solucion_candidata[viaje_a][pos_a])

                # Verificar capacidad en todos los viajes afectados.
                if not all(capacidad_respetada(v, instancia)
                           for v in solucion_candidata):
                    continue

                # Reordenar viajes.
                solucion_candidata = ordenar_viajes_de_forma_optima(
                    solucion_candidata, instancia)

                # Evaluar.
                valor_candidato, factible = evaluar_solucion(
                    solucion_candidata, instancia)
                if factible and valor_candidato < mejor_valor - 1e-7:
                    solucion_actual = [
                        list(v) for v in solucion_candidata]
                    mejor_valor = valor_candidato
                    hubo_mejora = True
                    break
            if hubo_mejora:
                break

        if hubo_mejora:
            continue

        # ============================================================
        # VECINDARIO 3: 2-OPT INTRA-VIAJE
        # Idea: invertir un segmento [i..j] dentro de un mismo viaje.
        # No cambia capacidad (mismos clientes), pero puede acortar la
        # ruta y por tanto reducir la latencia.
        # ============================================================
        for indice_viaje in range(len(solucion_actual)):
            viaje = solucion_actual[indice_viaje]
            longitud_viaje = len(viaje)
            for i in range(longitud_viaje - 1):
                for j in range(i + 1, longitud_viaje):
                    # Construir nuevo viaje con segmento invertido.
                    nuevo_viaje = (
                        viaje[:i]
                        + viaje[i:j + 1][::-1]   # segmento invertido
                        + viaje[j + 1:]
                    )

                    solucion_candidata = [list(v) for v in solucion_actual]
                    solucion_candidata[indice_viaje] = nuevo_viaje

                    # No hace falta revisar capacidad (no cambio carga).
                    solucion_candidata = ordenar_viajes_de_forma_optima(
                        solucion_candidata, instancia)

                    valor_candidato, factible = evaluar_solucion(
                        solucion_candidata, instancia)
                    if factible and valor_candidato < mejor_valor - 1e-7:
                        solucion_actual = [
                            list(v) for v in solucion_candidata]
                        mejor_valor = valor_candidato
                        hubo_mejora = True
                        break
                if hubo_mejora:
                    break
            if hubo_mejora:
                break
        # Si ninguno de los 3 vecindarios mejoro, hubo_mejora sigue en
        # False y el while termina: hemos llegado a un optimo local VND.

    return solucion_actual, mejor_valor


# ===========================================================================
# 7. METAHEURISTICA GRASP CON MULTIARRANQUE
# ===========================================================================

def metaheuristica_grasp(instancia, maximo_iteraciones=100, alpha=0.3,
                          modo_construccion="vecino_mas_cercano",
                          semilla=12345, limite_tiempo=None,
                          mostrar_progreso=False):
    """
    Algoritmo GRASP completo:
        1. Multiples iteraciones (multiarranque).
        2. En cada iteracion: construir solucion aleatorizada + buscar
           localmente con VND.
        3. Quedarse con la mejor solucion encontrada.

    ENTRADA:
        instancia          : objeto Instancia.
        maximo_iteraciones : int, numero maximo de iteraciones GRASP.
        alpha              : float, controla la aleatoriedad en la RCL.
        modo_construccion  : str, criterio de la fase golosa.
        semilla            : int, para reproducibilidad del RNG.
        limite_tiempo      : float opcional, segundos como tope.
        mostrar_progreso   : bool, imprime nuevos mejores si True.

    SALIDA:
        diccionario con:
            "solucion"     : la mejor solucion encontrada.
            "objetivo"     : valor objetivo de esa solucion.
            "iteraciones"  : cuantas iteraciones se hicieron.
            "tiempo"       : segundos totales.
            "historial"    : lista con el mejor valor visto en cada iter.
    """
    # Generador aleatorio reproducible.
    generador_aleatorio = random.Random(semilla)

    mejor_solucion_global = None
    mejor_valor_global = float("inf")

    tiempo_inicial = time.time()
    historial_valores = []

    for iteracion in range(maximo_iteraciones):
        # Chequear limite de tiempo.
        if (limite_tiempo is not None
                and (time.time() - tiempo_inicial) >= limite_tiempo):
            break

        # FASE 1: construccion golosa aleatorizada.
        solucion_inicial = construir_solucion_grasp(
            instancia, alpha, generador_aleatorio, modo=modo_construccion)

        # FASE 2: busqueda local VND.
        solucion_mejorada, valor_mejorado = busqueda_local_vnd(
            solucion_inicial, instancia)

        # FASE 3: actualizar el mejor global si corresponde.
        if valor_mejorado < mejor_valor_global:
            mejor_valor_global = valor_mejorado
            mejor_solucion_global = [list(v) for v in solucion_mejorada]
            if mostrar_progreso:
                print(f"  iter {iteracion:4d}: "
                      f"nuevo mejor = {mejor_valor_global:.2f}")

        historial_valores.append(mejor_valor_global)

    tiempo_total = time.time() - tiempo_inicial

    return {
        "solucion": mejor_solucion_global,
        "objetivo": mejor_valor_global,
        "iteraciones": len(historial_valores),
        "tiempo": tiempo_total,
        "historial": historial_valores,
    }


# ===========================================================================
# 8. UTILIDADES DE REPORTE
# ===========================================================================

def imprimir_solucion(resultado, instancia):
    """
    Imprime de forma amigable la solucion encontrada por GRASP.

    ENTRADA:
        resultado : diccionario devuelto por metaheuristica_grasp.
        instancia : objeto Instancia.

    SALIDA:
        Nada; imprime por consola.
    """
    solucion = resultado["solucion"]
    viajes = [v for v in solucion if v]

    print(f"\n--- Mejor solucion para la instancia '{instancia.nombre}' ---")
    print(f"Funcion objetivo (suma de latencias): "
          f"{resultado['objetivo']:.2f}")

    info_extra = ""
    if instancia.num_viajes_sugerido is not None:
        info_extra = (f"  (nbTrips indicado = "
                      f"{instancia.num_viajes_sugerido}, "
                      f"minimo por capacidad = {instancia.viajes_minimos})")
    print(f"Numero de viajes usados: {len(viajes)}{info_extra}")
    print(f"Iteraciones GRASP: {resultado['iteraciones']}   "
          f"Tiempo: {resultado['tiempo']:.2f} s")

    tiempo_jornada_total = 0.0
    for numero_viaje, viaje in enumerate(viajes, start=1):
        carga = sum(instancia.demandas[c] for c in viaje)
        duracion = calcular_duracion_viaje(viaje, instancia)
        tiempo_jornada_total += duracion
        descripcion_ruta = ("0 -> "
                            + " -> ".join(str(c) for c in viaje)
                            + " -> 0")
        print(f"  Viaje {numero_viaje}: {descripcion_ruta}")
        print(f"           carga={carga:.0f}/"
              f"{instancia.capacidad_vehiculo:.0f}  "
              f"duracion={duracion:.2f}")

    if instancia.tiempo_max_jornada is not None:
        print(f"  Tiempo total de jornada: "
              f"{tiempo_jornada_total:.2f} / "
              f"T={instancia.tiempo_max_jornada}")

    # Verificacion: la suma de latencias individuales debe coincidir
    # con el objetivo reportado (si no hay penalizaciones).
    suma_verificacion = sum(
        calcular_latencia_por_cliente(solucion, instancia).values())
    print(f"  Verificacion (suma de latencias recalculada): "
          f"{suma_verificacion:.2f}")


def ejecutar_experimento(instancia, configuraciones, replicas=10,
                          semilla_base=1000, limite_tiempo=None):
    """
    Ejecuta varias CONFIGURACIONES (combinaciones de parametros) sobre
    UNA instancia, con REPLICAS para estadistica, y compara resultados.

    ENTRADA:
        instancia        : objeto Instancia.
        configuraciones  : lista de dicts, cada uno con keys "name",
                           "mode", "alpha", "max_iter".
        replicas         : int, cuantas corridas por configuracion.
        semilla_base     : int, semilla inicial; cada replica usa
                           semilla_base + r.
        limite_tiempo    : float opcional, tope por corrida.

    SALIDA:
        tupla (filas, mejor_resultado_global, nombre_mejor_config).
        - filas                 : lista con estadisticas por config.
        - mejor_resultado_global: dict tipo el de metaheuristica_grasp.
        - nombre_mejor_config   : str con el name de la mejor config.
    """
    print(f"\n===== Experimento: '{instancia.nombre}'  "
          f"(n={instancia.numero_clientes}, "
          f"Q={instancia.capacidad_vehiculo}, "
          f"T={instancia.tiempo_max_jornada}, "
          f"nbTrips={instancia.num_viajes_sugerido}) =====")

    filas_resumen = []
    mejor_resultado_global = None
    mejor_valor_global = float("inf")
    nombre_mejor_config = None

    for config in configuraciones:
        valores_obtenidos = []
        tiempos_obtenidos = []
        mejor_resultado_config = None
        mejor_valor_config = float("inf")

        for r in range(replicas):
            resultado = metaheuristica_grasp(
                instancia,
                maximo_iteraciones=config["max_iter"],
                alpha=config["alpha"],
                modo_construccion=config["mode"],
                semilla=semilla_base + r,
                limite_tiempo=limite_tiempo,
            )
            valores_obtenidos.append(resultado["objetivo"])
            tiempos_obtenidos.append(resultado["tiempo"])
            if resultado["objetivo"] < mejor_valor_config:
                mejor_valor_config = resultado["objetivo"]
                mejor_resultado_config = resultado

        filas_resumen.append({
            "config": config["name"],
            "mejor": min(valores_obtenidos),
            "promedio": sum(valores_obtenidos) / len(valores_obtenidos),
            "peor": max(valores_obtenidos),
            "tiempo_promedio": (sum(tiempos_obtenidos)
                                / len(tiempos_obtenidos)),
            "replicas": replicas,
        })

        if mejor_valor_config < mejor_valor_global:
            mejor_valor_global = mejor_valor_config
            mejor_resultado_global = mejor_resultado_config
            nombre_mejor_config = config["name"]

    # Tabla comparativa.
    print(f"\n{'Config':<14}{'Mejor':>12}{'Promedio':>12}"
          f"{'Peor':>12}{'T.prom(s)':>12}{'Repl.':>8}")
    print("-" * 70)
    for fila in filas_resumen:
        print(f"{fila['config']:<14}{fila['mejor']:>12.2f}"
              f"{fila['promedio']:>12.2f}{fila['peor']:>12.2f}"
              f"{fila['tiempo_promedio']:>12.3f}"
              f"{fila['replicas']:>8d}")
    print(f"\nMejor configuracion: {nombre_mejor_config}")
    imprimir_solucion(mejor_resultado_global, instancia)

    return filas_resumen, mejor_resultado_global, nombre_mejor_config


# ===========================================================================
# 9. AUTO-PRUEBA DE VALIDACION
# ===========================================================================

def auto_prueba():
    """
    Prueba interna que verifica que las formulas de latencia funcionen
    correctamente, comparandolas con calculos manuales.

    Si hay un error en alguna formula, las asserts fallaran y el
    programa se detendra antes de procesar instancias reales.
    """
    # Matriz simetrica simple de 4 nodos.
    matriz_prueba = [
        [0, 10, 20, 30],
        [10, 0, 12, 22],
        [20, 12, 0, 15],
        [30, 22, 15, 0],
    ]

    # ----- Test 1: latencia interna sin tiempos de servicio -----
    instancia_test = Instancia(
        nombre="test1", demandas=[0, 5, 5, 5],
        capacidad_vehiculo=20, matriz_distancias=matriz_prueba)
    viaje_test = [1, 2, 3]
    # Manualmente, las latencias son:
    #   cliente 1: 10
    #   cliente 2: 10 + 12 = 22
    #   cliente 3: 10 + 12 + 15 = 37
    # suma = 69
    valor_manual = (matriz_prueba[0][1]
                    + matriz_prueba[0][1] + matriz_prueba[1][2]
                    + matriz_prueba[0][1] + matriz_prueba[1][2]
                    + matriz_prueba[2][3])
    assert abs(calcular_latencia_interna_viaje(viaje_test, instancia_test)
               - valor_manual) < 1e-9

    # ----- Test 2: solucion con 2 viajes (verifica offsets) -----
    solucion_test = [[1, 2], [3]]
    duracion_viaje_1 = (matriz_prueba[0][1]
                        + matriz_prueba[1][2]
                        + matriz_prueba[2][0])
    # Latencia esperada:
    #   viaje 1: 10 + (10+12) = 32
    #   viaje 2: cliente 3 espera la duracion del viaje 1, luego viaja 30
    valor_manual_2 = ((matriz_prueba[0][1]
                       + (matriz_prueba[0][1] + matriz_prueba[1][2]))
                      + (duracion_viaje_1 + matriz_prueba[0][3]))
    valor_calculado, factible = evaluar_solucion(solucion_test,
                                                  instancia_test)
    assert factible and abs(valor_calculado - valor_manual_2) < 1e-9

    # ----- Test 3: con tiempos de servicio (cliente 1 tarda 7) -----
    instancia_test_servicios = Instancia(
        nombre="test2", demandas=[0, 5, 5],
        capacidad_vehiculo=20, matriz_distancias=matriz_prueba,
        tiempos_servicio=[0, 7, 0])
    # Latencia del cliente 1: 10
    # Latencia del cliente 2: 10 + 7 (servicio del 1) + 12 = 29
    # suma = 39
    valor_esperado_servicios = 10 + (10 + 7 + 12)
    assert abs(calcular_latencia_interna_viaje(
        [1, 2], instancia_test_servicios) - valor_esperado_servicios) < 1e-9

    print("Auto-prueba de latencia (con y sin servicio): OK")


# ===========================================================================
# 10. PUNTO DE ENTRADA / LINEA DE COMANDOS
# ===========================================================================

# Configuraciones por defecto a probar en los experimentos.
# Cada una varia el modo de construccion (mode), el alpha de la RCL
# y el numero maximo de iteraciones (max_iter).
CONFIGURACIONES_POR_DEFECTO = [
    {"name": "C1-nn-a02",  "mode": "vecino_mas_cercano",
     "alpha": 0.2, "max_iter": 80},
    {"name": "C2-nn-a04",  "mode": "vecino_mas_cercano",
     "alpha": 0.4, "max_iter": 80},
    {"name": "C3-ins-a03", "mode": "menor_incremento",
     "alpha": 0.3, "max_iter": 80},
    {"name": "C4-nn-a00",  "mode": "vecino_mas_cercano",
     "alpha": 0.0, "max_iter": 80},
]


def main():
    """
    Punto de entrada del programa.
    Procesa argumentos de linea de comandos y lanza los experimentos.

    Ejemplo de uso:
        python3 mtvrp_grasp_comentado.py MT-DMP10s0-01.txt VRPNC1m.TXT
        python3 mtvrp_grasp_comentado.py --replicas 5 instancia.txt
        python3 mtvrp_grasp_comentado.py --selftest
    """
    parser = argparse.ArgumentParser(
        description="GRASP para MTVRP con minima latencia.")
    parser.add_argument("instancias", nargs="*",
                        help="Rutas a archivos de instancia.")
    parser.add_argument("--replicas", type=int, default=10,
                        help="Replicas por configuracion (default: 10).")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Limite de tiempo en segundos por corrida.")
    parser.add_argument("--round", action="store_true",
                        help="Redondear distancias euclidianas.")
    parser.add_argument("--selftest", action="store_true",
                        help="Solo ejecutar la auto-prueba y salir.")
    argumentos = parser.parse_args()

    # Si piden solo la auto-prueba, salimos despues de ella.
    if argumentos.selftest:
        auto_prueba()
        return

    # Ejecutamos siempre la auto-prueba antes de procesar instancias.
    auto_prueba()

    if not argumentos.instancias:
        print("Indica al menos un archivo de instancia. Ejemplo:")
        print("  python3 mtvrp_grasp_comentado.py "
              "MT-DMP10s0-01.txt VRPNC1m.TXT")
        return

    for ruta in argumentos.instancias:
        if not os.path.isfile(ruta):
            print(f"Aviso: no se encontro '{ruta}', se omite.")
            continue
        instancia = leer_instancia(ruta, redondear=argumentos.round)
        print(instancia)
        ejecutar_experimento(
            instancia,
            CONFIGURACIONES_POR_DEFECTO,
            replicas=argumentos.replicas,
            limite_tiempo=argumentos.time_limit,
        )


if __name__ == "__main__":
    main()