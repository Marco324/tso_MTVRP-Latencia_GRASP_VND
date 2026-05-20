"""
===========================================================================
MTVRP con funcion objetivo de MINIMA LATENCIA
Algoritmo heuristico: GRASP + Busqueda Local (VND) + Multiarranque
===========================================================================

Proyecto Integrador de Aprendizaje - Temas Selectos de Optimizacion

Problema:
    Un unico vehiculo realiza MULTIPLES VIAJES desde un deposito para
    atender a todos los clientes. Cada viaje empieza y termina en el
    deposito. Restricciones: capacidad del vehiculo Q por viaje y, cuando
    se proporcione, tiempo maximo de jornada T para la suma de las
    duraciones de todos los viajes.

Objetivo:
    Minimizar la SUMA DE LAS LATENCIAS de los clientes, es decir, la suma
    de los tiempos que cada cliente espera desde que el vehiculo sale por
    primera vez del deposito hasta que ese cliente recibe el servicio.

Idea central de la latencia con multiples viajes:
    Para un viaje k, todos sus clientes "arrastran" como espera fija la
    suma de las duraciones COMPLETAS de los viajes anteriores (incluido
    el arco de regreso al deposito y los tiempos de servicio). Dentro del
    viaje, la latencia de cada cliente es su tiempo de llegada acumulado.

        L = sum_{k} [ n_k * offset_k + (latencia interna del viaje k) ]

    donde offset_k = suma de duraciones completas de los viajes 1..k-1.
    El tiempo de servicio de un cliente NO cuenta en su propia latencia
    (se mide al llegar), pero SI retrasa la llegada a los clientes
    posteriores.

Resultado teorico aprovechado:
    Dado un conjunto FIJO de viajes, el orden optimo entre viajes se
    obtiene ordenandolos de forma creciente por la razon
    (duracion del viaje) / (numero de clientes del viaje).
    Se demuestra con un argumento de intercambio entre viajes adyacentes.

Formatos de instancia soportados:
    (1) Con MATRIZ explicita (TravelTimes) - p.ej. MT-DMP10s0-01
    (2) Con COORDENADAS (CoorX/CoorY)       - p.ej. VRPNC1m

===========================================================================
"""

import math
import random
import time
import argparse
import os
import multiprocessing
from functools import partial


# ===========================================================================
# 1. REPRESENTACION DE LA INSTANCIA
# ===========================================================================

# Se crea la clase Instance, la cual va ser el esqueleto de las instancias ejemplo de la ingeniera
class Instance:
    """Contenedor de una instancia del MTVRP (deposito = indice 0)."""

    def __init__(self, name, demand, Q, dist, service=None, T=None,
                 nb_trips=None):
        self.name = name
        self.demand = demand
        self.n = len(demand) - 1
        self.depot = 0
        self.Q = Q
        self.T = T
        self.nb_trips = nb_trips
        self.dist = dist
        self.service = service if service is not None else [0.0] * len(demand)
        self.total_demand = sum(demand)
        self.min_trips = math.ceil(self.total_demand / Q) if Q else None

    def __repr__(self):
        return (f"Instance(name={self.name}, n={self.n}, Q={self.Q}, "
                f"T={self.T}, nb_trips={self.nb_trips}, "
                f"demanda_total={self.total_demand:.0f})")


# ===========================================================================
# 2. LECTURA DE INSTANCIAS
# ===========================================================================

# Función para pasar cordenadas (X,Y) a una matriz de TravelTime, utilizando la ecuación de distancia entre dos puntos
def _euclidean_matrix(coords, round_dist=False):
    m = len(coords)
    d = [[0.0] * m for _ in range(m)]
    for i in range(m):
        xi, yi = coords[i]
        for j in range(m):
            xj, yj = coords[j]
            v = math.hypot(xi - xj, yi - yj)
            d[i][j] = float(round(v)) if round_dist else v
    return d


def _num_after(line):
    """Ultimo token numerico de una linea tipo 'clave: valor'."""
    return line.replace(":", " ").split()[-1]


def _read_official(lines, name, round_dist=False):
    """
    Lector de los formatos OFICIALES del proyecto.

    Claves reconocidas (con o sin ':'):
        nbClients, nbTrips, VehCapacity, MaxTime (opcional)
        ClientDemands:  -> n valores (clientes 1..n)
        ServiceTimes:   -> n valores (clientes 1..n)
        TravelTimes:    -> matriz (n+1)x(n+1), nodo 0 = deposito
        CoorX CoorY     -> (n+1) pares; el primero es el deposito
    Los numeros de cada seccion pueden venir en una o varias lineas.
    """
    n = nb_trips = None
    Q = T = None
    demands_raw, service_raw, matrix_raw, coords_raw = [], [], [], []
    section = None

    for ln in lines:
        low = ln.lower()
        if low.startswith("nbclients"):
            n = int(float(_num_after(ln)));        section = None
        elif low.startswith("nbtrips"):
            nb_trips = int(float(_num_after(ln)));  section = None
        elif low.startswith("vehcapacity"):
            Q = float(_num_after(ln));              section = None
        elif low.startswith(("maxtime", "max_time", "maxduration",
                              "jornada")):
            T = float(_num_after(ln));              section = None
        elif low.startswith("clientdemands"):
            section = "dem"
        elif low.startswith("servicetimes"):
            section = "srv"
        elif low.startswith("traveltimes"):
            section = "mat"
        elif low.startswith("coorx") or ("coor" in low and "coory" in low):
            section = "coor"
        else:
            toks = ln.replace(",", " ").split()
            if not toks:
                continue
            try:
                nums = [float(t) for t in toks]
            except ValueError:
                continue
            if section == "dem":
                demands_raw += nums
            elif section == "srv":
                service_raw += nums
            elif section == "mat":
                matrix_raw += nums
            elif section == "coor":
                coords_raw += nums

    if n is None or Q is None:
        raise ValueError("Faltan nbClients o VehCapacity en la instancia")

    demand = [0.0] + demands_raw[:n]
    service = ([0.0] + service_raw[:n]) if service_raw else [0.0] * (n + 1)

    if matrix_raw:
        size = n + 1
        if len(matrix_raw) < size * size:
            raise ValueError("La matriz TravelTimes esta incompleta")
        dist = [matrix_raw[i * size:(i + 1) * size] for i in range(size)]
    elif coords_raw:
        pts = [(coords_raw[2 * i], coords_raw[2 * i + 1])
               for i in range(n + 1)]
        dist = _euclidean_matrix(pts, round_dist)
    else:
        raise ValueError("No se encontro TravelTimes ni coordenadas")

    return Instance(name=name, demand=demand, Q=Q, dist=dist,
                    service=service, T=T, nb_trips=nb_trips)


def read_instance(path, round_dist=False):
    """
    Lector general con deteccion automatica de formato:
      - 'nbClients' presente            -> formato oficial del proyecto
      - 'NODE_COORD_SECTION'/'DEMAND'   -> CVRP / TSPLIB
      - en otro caso                    -> formato simple con seccion NODES
    """
    with open(path, "r") as f:
        raw = f.read().replace("\r", "")
    lines = [ln.strip() for ln in raw.split("\n")]
    lines_ne = [ln for ln in lines if ln != ""]
    name = os.path.splitext(os.path.basename(path))[0]
    blob = " ".join(ln.lower() for ln in lines_ne)

    if "nbclients" in blob:
        return _read_official(lines_ne, name, round_dist)

    if "node_coord_section" in blob or "demand_section" in blob:
        cap = tmax = None
        coords, demand = {}, {}
        depot_id = None
        section = None
        for ln in lines_ne:
            u = ln.upper()
            if u.startswith("NAME"):
                name = ln.split()[-1]
            elif u.startswith("CAPACITY"):
                cap = float(ln.replace(":", " ").split()[-1])
            elif u.startswith(("MAX_TIME", "DISTANCE", "MAX_DURATION")):
                tmax = float(ln.replace(":", " ").split()[-1])
            elif u.startswith("NODE_COORD_SECTION"):
                section = "coord"
            elif u.startswith("DEMAND_SECTION"):
                section = "demand"
            elif u.startswith("DEPOT_SECTION"):
                section = "depot"
            elif u.startswith(("EOF", "DISPLAY_DATA")):
                section = None
            elif u.startswith(("TYPE", "COMMENT", "DIMENSION",
                               "EDGE_WEIGHT")):
                continue
            else:
                tok = ln.split()
                if section == "coord" and len(tok) >= 3:
                    coords[int(tok[0])] = (float(tok[1]), float(tok[2]))
                elif section == "demand" and len(tok) >= 2:
                    demand[int(tok[0])] = float(tok[1])
                elif section == "depot":
                    v = int(tok[0])
                    if v != -1 and depot_id is None:
                        depot_id = v
        if depot_id is None:
            depot_id = min(coords)
        others = sorted(k for k in coords if k != depot_id)
        order = [depot_id] + others
        pts = [coords[o] for o in order]
        dem = [demand.get(o, 0.0) for o in order]
        dem[0] = 0.0
        return Instance(name=name, demand=dem, Q=cap,
                        dist=_euclidean_matrix(pts, round_dist), T=tmax)

    cap = tmax = None
    coords, demand = {}, {}
    section = None
    for ln in lines_ne:
        u = ln.upper()
        if u.startswith("NAME"):
            name = ln.split()[-1]
        elif u.startswith("CAPACITY"):
            cap = float(ln.split()[-1])
        elif u.startswith(("MAX_TIME", "MAX_DURATION", "JORNADA")):
            tmax = float(ln.split()[-1])
        elif u.startswith(("DIMENSION", "COMMENT", "TYPE")):
            continue
        elif u.startswith(("NODES", "NODE_SECTION", "DATA")):
            section = "nodes"
        else:
            tok = ln.split()
            if section == "nodes" and len(tok) >= 4:
                idx = int(tok[0])
                coords[idx] = (float(tok[1]), float(tok[2]))
                demand[idx] = float(tok[3])
    depot_id = min(coords)
    for k, v in demand.items():
        if v == 0:
            depot_id = k
            break
    others = sorted(k for k in coords if k != depot_id)
    order = [depot_id] + others
    pts = [coords[o] for o in order]
    dem = [demand.get(o, 0.0) for o in order]
    dem[0] = 0.0
    return Instance(name=name, demand=dem, Q=cap,
                    dist=_euclidean_matrix(pts, round_dist), T=tmax)


# ===========================================================================
# 3. EVALUACION (FUNCION OBJETIVO DE LATENCIA, CON TIEMPOS DE SERVICIO)
# ===========================================================================
#
# Representacion: solution = [viaje_1, ..., viaje_m]; cada viaje es una
# lista ordenada de clientes. El deposito (0) es implicito al inicio y al
# final de cada viaje.
# ===========================================================================

def trip_duration(trip, inst):
    """Duracion completa deposito -> ... -> deposito (incluye servicio)."""
    if not trip:
        return 0.0
    d, s = inst.dist, inst.service
    acc = 0.0
    prev = 0
    for node in trip:
        acc += d[prev][node] + s[node]
        prev = node
    acc += d[prev][0]
    return acc


def trip_internal_latency(trip, inst):
    """
    Suma de los tiempos de llegada de los clientes del viaje (sin offset).
    Con servicios nulos equivale a sum (n-i+1)*c[v_{i-1}][v_i].
    """
    if not trip:
        return 0.0
    d, s = inst.dist, inst.service
    acc = 0.0
    total = 0.0
    prev = 0
    for node in trip:
        acc += d[prev][node]
        total += acc
        acc += s[node]
        prev = node
    return total


def evaluate(solution, inst, penalty=1e9):
    """Devuelve (objetivo = suma de latencias, factible)."""
    trips = [t for t in solution if t]
    total = 0.0
    offset = 0.0
    feasible = True
    infeas = 0.0
    total_time = 0.0
    for trip in trips:
        load = sum(inst.demand[c] for c in trip)
        if load > inst.Q + 1e-9:
            feasible = False
            infeas += (load - inst.Q)
        n_k = len(trip)
        total += n_k * offset + trip_internal_latency(trip, inst)
        dur = trip_duration(trip, inst)
        offset += dur
        total_time += dur
    if inst.T is not None and total_time > inst.T + 1e-6:
        feasible = False
        infeas += (total_time - inst.T)
    if not feasible:
        total += penalty + infeas * penalty
    return total, feasible


def latency_per_customer(solution, inst):
    """Diccionario {cliente: latencia}. Util para validar el calculo."""
    d, s = inst.dist, inst.service
    res = {}
    offset = 0.0
    for trip in solution:
        if not trip:
            continue
        acc = 0.0
        prev = 0
        for node in trip:
            acc += d[prev][node]
            res[node] = offset + acc
            acc += s[node]
            prev = node
        offset += acc + d[prev][0]
    return res


# ===========================================================================
# 4. ORDEN OPTIMO ENTRE VIAJES (REGLA EXACTA DE INTERCAMBIO)
# ===========================================================================

def optimal_trip_order(solution, inst):
    """Ordena los viajes por (duracion / nº de clientes) ascendente."""
    trips = [t for t in solution if t]

    def ratio(t):
        nk = len(t)
        return trip_duration(t, inst) / nk if nk else float("inf")

    trips.sort(key=ratio)
    return trips

# ===========================================================================
# EVALUACION CON CACHE (Para optimizar la Busqueda Local a O(m log m))
# ===========================================================================

def get_trip_stats(trip, inst):
    """Calcula estadisticas basicas de un viaje en O(len(trip))."""
    if not trip: return 0.0, 0.0, 0.0
    load = sum(inst.demand[c] for c in trip)
    dur = trip_duration(trip, inst)
    lat = trip_internal_latency(trip, inst)
    return load, dur, lat

def evaluate_cached(stats_list, inst, penalty=1e9):
    """
    Evalua la solucion a partir de estadisticas cacheadas. 
    Evita recalcular rutas no modificadas. Complejidad O(m log m).
    """
    valid = [s for s in stats_list if s[0]]
    # Ordenar viajes por la regla de oro: duracion / numero de clientes
    valid.sort(key=lambda x: x[2] / len(x[0]))

    total = 0.0
    offset = 0.0
    feasible = True
    infeas = 0.0
    total_time = 0.0

    for trip, load, dur, lat in valid:
        if load > inst.Q + 1e-9:
            feasible = False
            infeas += (load - inst.Q)

        n_k = len(trip)
        total += n_k * offset + lat
        offset += dur
        total_time += dur

    if inst.T is not None and total_time > inst.T + 1e-6:
        feasible = False
        infeas += (total_time - inst.T)

    if not feasible:
        total += penalty + infeas * penalty

    return total, feasible


# ===========================================================================
# 5. CONSTRUCCION GOLOSA ALEATORIZADA (FASE DE CONSTRUCCION DE GRASP)
# ===========================================================================

def construct_grasp(inst, alpha, rng, mode="nn"):
    """
    Construccion golosa aleatorizada (RCL parametrizada por alpha).
        alpha = 0 -> goloso puro ; alpha = 1 -> aleatorio
    mode = "nn"  cliente mas cercano al nodo actual
           "ins" menor incremento de llegada al anexar al final
    """
    d = inst.dist
    unvisited = set(range(1, inst.n + 1))
    solution = []
    while unvisited:
        trip = []
        cap_left = inst.Q
        current = 0
        while True:
            cand = [c for c in unvisited
                    if inst.demand[c] <= cap_left + 1e-9]
            if not cand:
                break
            scored = sorted((d[current][c], c) for c in cand)
            gmin, gmax = scored[0][0], scored[-1][0]
            thr = gmin + alpha * (gmax - gmin)
            rcl = [c for (v, c) in scored if v <= thr + 1e-9]
            chosen = rng.choice(rcl)
            trip.append(chosen)
            unvisited.discard(chosen)
            cap_left -= inst.demand[chosen]
            current = chosen
        if not trip:
            raise ValueError("Existe un cliente con demanda mayor que Q")
        solution.append(trip)
    return optimal_trip_order(solution, inst)


# ===========================================================================
# 6. BUSQUEDA LOCAL (VND: relocate, swap, 2-opt) + REORDEN DE VIAJES (OPTIMIZADA)
# ===========================================================================

def local_search(solution, inst):
    """Busqueda local VND usando evaluacion en caché O(m log m)."""
    # sol_stats es una lista de tuplas: (trip_list, load, duracion, latencia_interna)
    sol_stats = [(list(t), *get_trip_stats(t, inst)) for t in solution if t]
    best_val, _ = evaluate_cached(sol_stats, inst)

    improved = True
    while improved:
        improved = False

        # Vecindario 1: relocate
        for ti in range(len(sol_stats)):
            for pi in range(len(sol_stats[ti][0])):
                cust = sol_stats[ti][0][pi]
                for tj in range(len(sol_stats)):
                    for pj in range(len(sol_stats[tj][0]) + 1):
                        if ti == tj and (pj == pi or pj == pi + 1):
                            continue

                        trip_i = sol_stats[ti][0][:]
                        trip_i.pop(pi)
                        trip_j = sol_stats[tj][0][:] if ti != tj else trip_i

                        if ti == tj and pj > pi:
                            trip_j.insert(pj - 1, cust)
                        elif ti != tj:
                            trip_j.insert(pj, cust)
                        else:
                            trip_j.insert(pj, cust)

                        # Calculo de carga rápido (corte antes de evaluar heuristica)
                        load_i = sum(inst.demand[c] for c in trip_i)
                        load_j = sum(inst.demand[c] for c in trip_j) if ti != tj else load_i

                        if load_i > inst.Q + 1e-9 or (ti != tj and load_j > inst.Q + 1e-9):
                            continue

                        # Clonar caché y sustituir solo los 1 o 2 viajes afectados
                        cand = sol_stats[:]
                        cand[ti] = (trip_i, load_i, trip_duration(trip_i, inst), trip_internal_latency(trip_i, inst))
                        if ti != tj:
                            cand[tj] = (trip_j, load_j, trip_duration(trip_j, inst), trip_internal_latency(trip_j, inst))

                        val, feas = evaluate_cached(cand, inst)
                        if feas and val < best_val - 1e-7:
                            sol_stats = cand
                            best_val = val
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
        if improved: continue

        # Vecindario 2: swap
        flat = [(ti, pi) for ti in range(len(sol_stats)) for pi in range(len(sol_stats[ti][0]))]
        for a in range(len(flat)):
            for b in range(a + 1, len(flat)):
                ti, pi = flat[a]
                tj, pj = flat[b]

                trip_i = sol_stats[ti][0][:]
                trip_j = sol_stats[tj][0][:] if ti != tj else trip_i

                # Realizar el intercambio
                trip_i[pi], trip_j[pj] = trip_j[pj], trip_i[pi]

                load_i = sum(inst.demand[c] for c in trip_i)
                load_j = sum(inst.demand[c] for c in trip_j) if ti != tj else load_i

                if load_i > inst.Q + 1e-9 or (ti != tj and load_j > inst.Q + 1e-9):
                    continue

                cand = sol_stats[:]
                cand[ti] = (trip_i, load_i, trip_duration(trip_i, inst), trip_internal_latency(trip_i, inst))
                if ti != tj:
                    cand[tj] = (trip_j, load_j, trip_duration(trip_j, inst), trip_internal_latency(trip_j, inst))

                val, feas = evaluate_cached(cand, inst)
                if feas and val < best_val - 1e-7:
                    sol_stats = cand
                    best_val = val
                    improved = True
                    break
            if improved: break
        if improved: continue

        # Vecindario 3: 2-opt intra-viaje
        for ti in range(len(sol_stats)):
            trip = sol_stats[ti][0]
            L = len(trip)
            for i in range(L - 1):
                for j in range(i + 1, L):
                    nt = trip[:i] + trip[i:j + 1][::-1] + trip[j + 1:]
                    # Un 2-opt NUNCA altera la carga, reutilizamos la del caché
                    load_i = sol_stats[ti][1]

                    cand = sol_stats[:]
                    cand[ti] = (nt, load_i, trip_duration(nt, inst), trip_internal_latency(nt, inst))

                    val, feas = evaluate_cached(cand, inst)
                    if feas and val < best_val - 1e-7:
                        sol_stats = cand
                        best_val = val
                        improved = True
                        break
                if improved: break
            if improved: break

    # Al finalizar, reordenamos formalmente la lista final para retornarla
    final_sol = optimal_trip_order([s[0] for s in sol_stats], inst)
    return final_sol, best_val


# ===========================================================================
# 7. METAHEURISTICA GRASP CON MULTIARRANQUE (PARALELIZADO)
# ===========================================================================

def _grasp_worker(seed, inst, alpha, mode):
    """Worker independiente sin estado global para ejecutar en multiples hilos."""
    rng = random.Random(seed)
    init = construct_grasp(inst, alpha, rng, mode=mode)
    sol, val = local_search(init, inst)
    return sol, val

def grasp(inst, max_iter=100, alpha=0.3, mode="nn", seed=12345,
          time_limit=None, verbose=False):
    t0 = time.time()
    best_sol, best_val = None, float("inf")
    history = []

    # Preparar el pool y fijar las semillas para replicabilidad
    seeds = [seed + i for i in range(max_iter)]
    worker = partial(_grasp_worker, inst=inst, alpha=alpha, mode=mode)
    num_cores = multiprocessing.cpu_count()

    # Ejecutar en Pool usando todos los nucleos logicos
    with multiprocessing.Pool(processes=num_cores) as pool:
        for sol, val in pool.imap_unordered(worker, seeds):
            # Abortar ejecucion por limite de tiempo de manera segura
            if time_limit is not None and (time.time() - t0) >= time_limit:
                break
            
            if val < best_val:
                best_val = val
                best_sol = sol
                if verbose:
                    print(f"  nuevo mejor = {best_val:.2f}")
            history.append(best_val)

    return {
        "solution": best_sol,
        "objective": best_val,
        "iterations": len(history),
        "time": time.time() - t0,
        "history": history,
    }

# ===========================================================================
# 8. UTILIDADES DE REPORTE
# ===========================================================================

def print_solution(result, inst):
    sol = result["solution"]
    trips = [t for t in sol if t]
    print(f"\n--- Mejor solucion para la instancia '{inst.name}' ---")
    print(f"Funcion objetivo (suma de latencias): {result['objective']:.2f}")
    extra = ""
    if inst.nb_trips is not None:
        extra = (f"  (nbTrips indicado = {inst.nb_trips}, "
                 f"minimo por capacidad = {inst.min_trips})")
    print(f"Numero de viajes usados: {len(trips)}{extra}")
    print(f"Iteraciones GRASP: {result['iterations']}   "
          f"Tiempo: {result['time']:.2f} s")
    total_time = 0.0
    for k, trip in enumerate(trips, start=1):
        load = sum(inst.demand[c] for c in trip)
        dur = trip_duration(trip, inst)
        total_time += dur
        ruta = "0 -> " + " -> ".join(str(c) for c in trip) + " -> 0"
        print(f"  Viaje {k}: {ruta}")
        print(f"           carga={load:.0f}/{inst.Q:.0f}  "
              f"duracion={dur:.2f}")
    if inst.T is not None:
        print(f"  Tiempo total de jornada: {total_time:.2f} / T={inst.T}")
    suma = sum(latency_per_customer(sol, inst).values())
    print(f"  Verificacion (suma de latencias recalculada): {suma:.2f}")


def run_experiment(inst, configs, replicas=10, base_seed=1000,
                   time_limit=None):
    print(f"\n===== Experimento: '{inst.name}'  (n={inst.n}, Q={inst.Q}, "
          f"T={inst.T}, nbTrips={inst.nb_trips}) =====")
    rows = []
    overall_best, overall_best_val, overall_cfg = None, float("inf"), None
    for cfg in configs:
        vals, times = [], []
        cfg_best, cfg_best_val = None, float("inf")
        for r in range(replicas):
            res = grasp(inst, max_iter=cfg["max_iter"], alpha=cfg["alpha"],
                        mode=cfg["mode"], seed=base_seed + r,
                        time_limit=time_limit)
            vals.append(res["objective"])
            times.append(res["time"])
            if res["objective"] < cfg_best_val:
                cfg_best_val, cfg_best = res["objective"], res
        rows.append({
            "config": cfg["name"], "best": min(vals),
            "avg": sum(vals) / len(vals), "worst": max(vals),
            "avg_time": sum(times) / len(times), "replicas": replicas,
        })
        if cfg_best_val < overall_best_val:
            overall_best_val, overall_best = cfg_best_val, cfg_best
            overall_cfg = cfg["name"]
    print(f"\n{'Config':<14}{'Mejor':>12}{'Promedio':>12}"
          f"{'Peor':>12}{'T.prom(s)':>12}{'Repl.':>8}")
    print("-" * 70)
    for row in rows:
        print(f"{row['config']:<14}{row['best']:>12.2f}"
              f"{row['avg']:>12.2f}{row['worst']:>12.2f}"
              f"{row['avg_time']:>12.3f}{row['replicas']:>8d}")
    print(f"\nMejor configuracion: {overall_cfg}")
    print_solution(overall_best, inst)
    return rows, overall_best, overall_cfg


# ===========================================================================
# 9. AUTO-PRUEBA DE VALIDACION
# ===========================================================================

def _self_test():
    dist = [
        [0, 10, 20, 30],
        [10, 0, 12, 22],
        [20, 12, 0, 15],
        [30, 22, 15, 0],
    ]
    inst = Instance(name="t", demand=[0, 5, 5, 5], Q=20, dist=dist)
    trip = [1, 2, 3]
    manual = (dist[0][1]
              + dist[0][1] + dist[1][2]
              + dist[0][1] + dist[1][2] + dist[2][3])
    assert abs(trip_internal_latency(trip, inst) - manual) < 1e-9
    sol = [[1, 2], [3]]
    dur1 = dist[0][1] + dist[1][2] + dist[2][0]
    manual2 = (dist[0][1] + (dist[0][1] + dist[1][2])) + (dur1 + dist[0][3])
    val, feas = evaluate(sol, inst)
    assert feas and abs(val - manual2) < 1e-9
    inst2 = Instance(name="t2", demand=[0, 5, 5], Q=20, dist=dist,
                     service=[0, 7, 0])
    assert abs(trip_internal_latency([1, 2], inst2)
               - (10 + (10 + 7 + 12))) < 1e-9
    print("Auto-prueba de latencia (con y sin servicio): OK")


# ===========================================================================
# 10. PUNTO DE ENTRADA / LINEA DE COMANDOS
# ===========================================================================

DEFAULT_CONFIGS = [
    {"name": "C1-nn-a02",  "mode": "nn",  "alpha": 0.2, "max_iter": 80},
    {"name": "C2-nn-a04",  "mode": "nn",  "alpha": 0.4, "max_iter": 80},
    {"name": "C3-ins-a03", "mode": "ins", "alpha": 0.3, "max_iter": 80},
    {"name": "C4-nn-a00",  "mode": "nn",  "alpha": 0.0, "max_iter": 80},
]


def main():
    ap = argparse.ArgumentParser(
        description="GRASP para MTVRP con minima latencia.")
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--round", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return
    _self_test()

    if not args.instances:
        print("Indica al menos un archivo de instancia. Ejemplo:")
        print("  python3 mtvrp_grasp.py MT-DMP10s0-01.txt VRPNC1m.TXT")
        return

    for path in args.instances:
        if not os.path.isfile(path):
            print(f"Aviso: no se encontro '{path}', se omite.")
            continue
        inst = read_instance(path, round_dist=args.round)
        print(inst)
        run_experiment(inst, DEFAULT_CONFIGS, replicas=args.replicas,
                        time_limit=args.time_limit)


if __name__ == "__main__":
    main()