"""
===========================================================================
MTVRP con funcion objetivo de MINIMA LATENCIA  -  VERSION OPTIMIZADA
Algoritmo heuristico: GRASP + Busqueda Local (VND) + Multiarranque
                    + Evaluacion incremental con cache por viaje
                    + Paralelismo con multiprocessing
===========================================================================

Cambios principales frente a la version original:

1) Evaluacion incremental.
   En lugar de recalcular trip_internal_latency y trip_duration para TODOS
   los viajes en cada intento de movimiento, mantenemos un cache
   (carga, duracion, latencia interna) por viaje. Cuando un movimiento
   cambia 1 o 2 viajes, solo se recalculan esos. La evaluacion final pasa
   a ser O(m log m) con m = numero de viajes (tipicamente 5-10).

2) Movimientos sin copias profundas.
   En la busqueda local guardamos solo los viajes afectados, aplicamos el
   movimiento "en sitio" y si no mejora hacemos rollback de esos viajes.

3) Poda por capacidad ANTES de mover.
   Para relocate/swap entre viajes distintos comprobamos primero si la
   nueva carga cabe usando los caches, sin construir el viaje siquiera.

4) Paralelismo con multiprocessing.
   La fase de multiarranque (configuraciones x replicas) se reparte en
   un Pool de procesos. Esquiva el GIL y escala con los nucleos. Tambien
   se puede pedir un solo proceso (--jobs 1) para depurar.

Uso tipico:
   python3 mtvrp_grasp_fast.py MT-DMP10s0-01.txt VRPNC1m.TXT --jobs 8

Si --jobs no se especifica (o es 0) se usan TODOS los nucleos disponibles.
===========================================================================
"""

import math
import random
import time
import argparse
import os
from multiprocessing import Pool, cpu_count


# ===========================================================================
# 1. REPRESENTACION DE LA INSTANCIA
# ===========================================================================

class Instance:
    """Contenedor de una instancia del MTVRP (deposito = indice 0)."""

    __slots__ = ('name', 'demand', 'n', 'depot', 'Q', 'T', 'nb_trips',
                 'dist', 'service', 'total_demand', 'min_trips')

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
# 2. LECTURA DE INSTANCIAS  (idéntica a la version original)
# ===========================================================================

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
    return line.replace(":", " ").split()[-1]


def _read_official(lines, name, round_dist=False):
    n = nb_trips = None
    Q = T = None
    demands_raw, service_raw, matrix_raw, coords_raw = [], [], [], []
    section = None

    for ln in lines:
        low = ln.lower()
        if low.startswith("nbclients"):
            n = int(float(_num_after(ln))); section = None
        elif low.startswith("nbtrips"):
            nb_trips = int(float(_num_after(ln))); section = None
        elif low.startswith("vehcapacity"):
            Q = float(_num_after(ln)); section = None
        elif low.startswith(("maxtime", "max_time", "maxduration",
                              "jornada")):
            T = float(_num_after(ln)); section = None
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
# 3. METRICAS POR VIAJE (NUCLEO DE LA EVALUACION INCREMENTAL)
# ===========================================================================
#
# Cada viaje se resume en TRES numeros:
#   load   : suma de demandas de los clientes del viaje
#   dur    : duracion total deposito -> ... -> deposito (incluye servicio)
#   lat    : suma de tiempos de LLEGADA de los clientes (sin offset entre
#            viajes y sin contar el servicio del propio cliente)
#
# Con estas metricas:
#   * la factibilidad de capacidad se chequea con `load` (1 comparacion)
#   * el aporte de un viaje a la latencia total es n_k*offset_k + lat_k
#   * el offset_k es la suma acumulada de duraciones de viajes anteriores
#     usando el orden optimo (ratio dur/n_clientes ascendente)
# ===========================================================================

def compute_trip_metrics(trip, inst):
    """Calcula (load, duration, internal_latency) en una sola pasada."""
    if not trip:
        return 0.0, 0.0, 0.0
    d = inst.dist
    s = inst.service
    dem = inst.demand
    load = 0.0
    acc = 0.0
    lat = 0.0
    prev = 0
    for node in trip:
        acc += d[prev][node]
        lat += acc
        acc += s[node]
        load += dem[node]
        prev = node
    duration = acc + d[prev][0]
    return load, duration, lat


def evaluate_metrics(trips, loads, durations, latencies, inst, penalty=1e9):
    """
    Evalua una solucion usando metricas YA calculadas por viaje.
    Aplica internamente el orden optimo de viajes (regla del ratio
    duracion / nº clientes) SIN reordenar las listas (no muta nada).
    Complejidad: O(m log m) con m = numero de viajes (muy pequeño).
    """
    nonempty = []
    for i in range(len(trips)):
        if trips[i]:
            nonempty.append((i, len(trips[i])))
    if not nonempty:
        return 0.0, True

    Q = inst.Q
    feasible = True
    infeas = 0.0
    for i, _ in nonempty:
        if loads[i] > Q + 1e-9:
            feasible = False
            infeas += loads[i] - Q

    # Orden optimo: ratio duracion/clientes ascendente
    nonempty.sort(key=lambda x: durations[x[0]] / x[1])

    total = 0.0
    offset = 0.0
    total_time = 0.0
    for i, n_k in nonempty:
        total += n_k * offset + latencies[i]
        offset += durations[i]
        total_time += durations[i]

    if inst.T is not None and total_time > inst.T + 1e-6:
        feasible = False
        infeas += total_time - inst.T

    if not feasible:
        total += penalty + infeas * penalty
    return total, feasible


# ----- Wrappers de compatibilidad con la API original -----

def trip_duration(trip, inst):
    return compute_trip_metrics(trip, inst)[1]


def trip_internal_latency(trip, inst):
    return compute_trip_metrics(trip, inst)[2]


def evaluate(solution, inst, penalty=1e9):
    """Wrapper compatible con la API original (recalcula metricas)."""
    trips = [t for t in solution if t]
    loads, durations, latencies = [], [], []
    for t in trips:
        L, D, La = compute_trip_metrics(t, inst)
        loads.append(L)
        durations.append(D)
        latencies.append(La)
    return evaluate_metrics(trips, loads, durations, latencies, inst,
                             penalty)


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
# 4. ORDEN OPTIMO ENTRE VIAJES
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
# 5. CONSTRUCCION GOLOSA ALEATORIZADA
# ===========================================================================

def construct_grasp(inst, alpha, rng, mode="nn"):
    """Construccion golosa aleatorizada (RCL parametrizada por alpha)."""
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
# 6. BUSQUEDA LOCAL VND CON EVALUACION INCREMENTAL
# ===========================================================================
#
# Estructura interna durante la busqueda:
#   trips[k]      : lista de clientes del viaje k (mutable)
#   loads[k]      : carga del viaje k (cacheada)
#   durations[k]  : duracion del viaje k (cacheada)
#   latencies[k]  : latencia interna del viaje k (cacheada)
#
# Cuando se prueba un movimiento:
#   1) se guarda el estado de los viajes afectados (1 o 2)
#   2) se aplica el movimiento (modificacion en sitio)
#   3) se recalculan las metricas solo de esos viajes
#   4) se llama a evaluate_metrics() -> O(m log m)
#   5) si NO mejora, se restaura el estado guardado
#
# Asi evitamos copiar la solucion completa y recalcular n_k veces.
# ===========================================================================

def local_search(initial_trips, inst):
    """VND: relocate -> swap -> 2-opt, con cache de metricas por viaje."""
    trips = [list(t) for t in initial_trips if t]
    loads, durations, latencies = [], [], []
    for t in trips:
        L, D, La = compute_trip_metrics(t, inst)
        loads.append(L)
        durations.append(D)
        latencies.append(La)
    best_val, _ = evaluate_metrics(trips, loads, durations, latencies, inst)

    Q = inst.Q
    dem = inst.demand

    improved = True
    while improved:
        improved = False

        # ----------------------------------------------------------------
        # Vecindario 1: RELOCATE (mover un cliente a otra posicion/viaje)
        # ----------------------------------------------------------------
        m = len(trips)
        for ti in range(m):
            if improved:
                break
            ni = len(trips[ti])
            for pi in range(ni):
                if improved:
                    break
                cust = trips[ti][pi]
                d_cust = dem[cust]
                for tj in range(m):
                    if improved:
                        break
                    # Poda: capacidad en viaje destino (si es distinto)
                    if tj != ti and loads[tj] + d_cust > Q + 1e-9:
                        continue
                    nj = len(trips[tj])
                    for pj in range(nj + 1):
                        if ti == tj and (pj == pi or pj == pi + 1):
                            continue

                        if ti == tj:
                            # Movimiento intra-viaje: 1 viaje afectado
                            new_ti = trips[ti][:pi] + trips[ti][pi + 1:]
                            adj = pj - 1 if pj > pi else pj
                            new_ti.insert(adj, cust)
                            o_t = trips[ti]
                            o_L, o_D, o_La = (loads[ti], durations[ti],
                                              latencies[ti])
                            L, D, La = compute_trip_metrics(new_ti, inst)
                            trips[ti] = new_ti
                            loads[ti] = L
                            durations[ti] = D
                            latencies[ti] = La
                            val, feas = evaluate_metrics(
                                trips, loads, durations, latencies, inst)
                            if feas and val < best_val - 1e-7:
                                best_val = val
                                improved = True
                                break
                            else:
                                trips[ti] = o_t
                                loads[ti] = o_L
                                durations[ti] = o_D
                                latencies[ti] = o_La
                        else:
                            # Movimiento inter-viaje: 2 viajes afectados
                            new_ti = trips[ti][:pi] + trips[ti][pi + 1:]
                            new_tj = (trips[tj][:pj] + [cust]
                                      + trips[tj][pj:])
                            o_ti, o_Li, o_Di, o_Lai = (
                                trips[ti], loads[ti],
                                durations[ti], latencies[ti])
                            o_tj, o_Lj, o_Dj, o_Laj = (
                                trips[tj], loads[tj],
                                durations[tj], latencies[tj])
                            Li, Di, Lai = compute_trip_metrics(new_ti, inst)
                            Lj, Dj, Laj = compute_trip_metrics(new_tj, inst)
                            trips[ti] = new_ti
                            loads[ti] = Li
                            durations[ti] = Di
                            latencies[ti] = Lai
                            trips[tj] = new_tj
                            loads[tj] = Lj
                            durations[tj] = Dj
                            latencies[tj] = Laj
                            val, feas = evaluate_metrics(
                                trips, loads, durations, latencies, inst)
                            if feas and val < best_val - 1e-7:
                                best_val = val
                                improved = True
                                break
                            else:
                                trips[ti] = o_ti
                                loads[ti] = o_Li
                                durations[ti] = o_Di
                                latencies[ti] = o_Lai
                                trips[tj] = o_tj
                                loads[tj] = o_Lj
                                durations[tj] = o_Dj
                                latencies[tj] = o_Laj

        if improved:
            # Limpiar viajes vacios (un relocate pudo dejar un viaje vacio)
            keep = [i for i in range(len(trips)) if trips[i]]
            if len(keep) != len(trips):
                trips = [trips[i] for i in keep]
                loads = [loads[i] for i in keep]
                durations = [durations[i] for i in keep]
                latencies = [latencies[i] for i in keep]
            continue

        # ----------------------------------------------------------------
        # Vecindario 2: SWAP (intercambiar dos clientes)
        # ----------------------------------------------------------------
        flat = [(ti, pi) for ti in range(len(trips))
                for pi in range(len(trips[ti]))]
        F = len(flat)
        for a in range(F):
            if improved:
                break
            ti, pi = flat[a]
            for b in range(a + 1, F):
                tj, pj = flat[b]
                cu = trips[ti][pi]
                cv = trips[tj][pj]
                # Poda por capacidad (solo si son viajes distintos)
                if ti != tj:
                    new_load_i = loads[ti] - dem[cu] + dem[cv]
                    new_load_j = loads[tj] - dem[cv] + dem[cu]
                    if (new_load_i > Q + 1e-9
                            or new_load_j > Q + 1e-9):
                        continue

                if ti == tj:
                    new_ti = trips[ti][:]
                    new_ti[pi], new_ti[pj] = new_ti[pj], new_ti[pi]
                    o_t = trips[ti]
                    o_L, o_D, o_La = (loads[ti], durations[ti],
                                      latencies[ti])
                    L, D, La = compute_trip_metrics(new_ti, inst)
                    trips[ti] = new_ti
                    loads[ti] = L
                    durations[ti] = D
                    latencies[ti] = La
                    val, feas = evaluate_metrics(
                        trips, loads, durations, latencies, inst)
                    if feas and val < best_val - 1e-7:
                        best_val = val
                        improved = True
                        break
                    else:
                        trips[ti] = o_t
                        loads[ti] = o_L
                        durations[ti] = o_D
                        latencies[ti] = o_La
                else:
                    new_ti = trips[ti][:]
                    new_tj = trips[tj][:]
                    new_ti[pi] = cv
                    new_tj[pj] = cu
                    o_ti, o_Li, o_Di, o_Lai = (
                        trips[ti], loads[ti],
                        durations[ti], latencies[ti])
                    o_tj, o_Lj, o_Dj, o_Laj = (
                        trips[tj], loads[tj],
                        durations[tj], latencies[tj])
                    Li, Di, Lai = compute_trip_metrics(new_ti, inst)
                    Lj, Dj, Laj = compute_trip_metrics(new_tj, inst)
                    trips[ti] = new_ti
                    loads[ti] = Li
                    durations[ti] = Di
                    latencies[ti] = Lai
                    trips[tj] = new_tj
                    loads[tj] = Lj
                    durations[tj] = Dj
                    latencies[tj] = Laj
                    val, feas = evaluate_metrics(
                        trips, loads, durations, latencies, inst)
                    if feas and val < best_val - 1e-7:
                        best_val = val
                        improved = True
                        break
                    else:
                        trips[ti] = o_ti
                        loads[ti] = o_Li
                        durations[ti] = o_Di
                        latencies[ti] = o_Lai
                        trips[tj] = o_tj
                        loads[tj] = o_Lj
                        durations[tj] = o_Dj
                        latencies[tj] = o_Laj

        if improved:
            continue

        # ----------------------------------------------------------------
        # Vecindario 3: 2-opt INTRA-VIAJE (invertir un segmento)
        # ----------------------------------------------------------------
        for ti in range(len(trips)):
            if improved:
                break
            trip = trips[ti]
            Lt = len(trip)
            for i in range(Lt - 1):
                if improved:
                    break
                for j in range(i + 1, Lt):
                    new_ti = (trip[:i] + trip[i:j + 1][::-1]
                              + trip[j + 1:])
                    o_t = trips[ti]
                    o_L, o_D, o_La = (loads[ti], durations[ti],
                                      latencies[ti])
                    L, D, La = compute_trip_metrics(new_ti, inst)
                    trips[ti] = new_ti
                    loads[ti] = L
                    durations[ti] = D
                    latencies[ti] = La
                    val, feas = evaluate_metrics(
                        trips, loads, durations, latencies, inst)
                    if feas and val < best_val - 1e-7:
                        best_val = val
                        improved = True
                        break
                    else:
                        trips[ti] = o_t
                        loads[ti] = o_L
                        durations[ti] = o_D
                        latencies[ti] = o_La

    # Devolver la solucion en su orden optimo
    nonempty = [(i, len(trips[i])) for i in range(len(trips))
                if trips[i]]
    nonempty.sort(key=lambda x: durations[x[0]] / x[1])
    final = [trips[i] for i, _ in nonempty]
    return final, best_val


# ===========================================================================
# 7. METAHEURISTICA GRASP CON MULTIARRANQUE
# ===========================================================================

def grasp(inst, max_iter=100, alpha=0.3, mode="nn", seed=12345,
          time_limit=None, verbose=False):
    rng = random.Random(seed)
    best_sol, best_val = None, float("inf")
    t0 = time.time()
    history = []
    for it in range(max_iter):
        if time_limit is not None and (time.time() - t0) >= time_limit:
            break
        init = construct_grasp(inst, alpha, rng, mode=mode)
        sol, val = local_search(init, inst)
        if val < best_val:
            best_val = val
            best_sol = [list(t) for t in sol]
            if verbose:
                print(f"  iter {it:4d}: nuevo mejor = {best_val:.2f}")
        history.append(best_val)
    return {
        "solution": best_sol,
        "objective": best_val,
        "iterations": len(history),
        "time": time.time() - t0,
        "history": history,
    }


# ===========================================================================
# 8. WRAPPERS PARA MULTIPROCESSING
# ===========================================================================
#
# Estrategia: cada proceso del Pool guarda la instancia en una variable
# global _WORKER_INSTANCE para no tener que pickearla en cada tarea.
# Cada tarea es una (config, semilla) y se resuelve con grasp(...).
# ===========================================================================

_WORKER_INSTANCE = None


def _init_worker(inst):
    global _WORKER_INSTANCE
    _WORKER_INSTANCE = inst


def _grasp_worker(args):
    cfg_name, max_iter, alpha, mode, seed, time_limit = args
    res = grasp(_WORKER_INSTANCE, max_iter=max_iter, alpha=alpha,
                mode=mode, seed=seed, time_limit=time_limit, verbose=False)
    res["config"] = cfg_name
    res["seed"] = seed
    return res


# ===========================================================================
# 9. UTILIDADES DE REPORTE
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
                   time_limit=None, jobs=1):
    print(f"\n===== Experimento: '{inst.name}'  (n={inst.n}, Q={inst.Q}, "
          f"T={inst.T}, nbTrips={inst.nb_trips}) =====")

    # Lista de tareas independientes: (cfg, replica) para todo el experimento
    tasks = []
    for cfg in configs:
        for r in range(replicas):
            tasks.append((cfg["name"], cfg["max_iter"], cfg["alpha"],
                          cfg["mode"], base_seed + r, time_limit))

    t_start = time.time()
    if jobs <= 1:
        # Camino secuencial (util para depurar)
        global _WORKER_INSTANCE
        _WORKER_INSTANCE = inst
        results = [_grasp_worker(t) for t in tasks]
    else:
        # Camino paralelo: Pool con instancia inicializada una sola vez
        with Pool(processes=jobs, initializer=_init_worker,
                  initargs=(inst,)) as pool:
            results = pool.map(_grasp_worker, tasks)
    t_total = time.time() - t_start

    # Agregar por configuracion
    rows = []
    overall_best_res = None
    overall_best_cfg = None
    for cfg in configs:
        cfg_results = [r for r in results if r["config"] == cfg["name"]]
        vals = [r["objective"] for r in cfg_results]
        times = [r["time"] for r in cfg_results]
        best = min(cfg_results, key=lambda r: r["objective"])
        rows.append({
            "config": cfg["name"], "best": min(vals),
            "avg": sum(vals) / len(vals), "worst": max(vals),
            "avg_time": sum(times) / len(times), "replicas": replicas,
        })
        if (overall_best_res is None
                or best["objective"] < overall_best_res["objective"]):
            overall_best_res = best
            overall_best_cfg = cfg["name"]

    print(f"\n{'Config':<14}{'Mejor':>12}{'Promedio':>12}"
          f"{'Peor':>12}{'T.prom(s)':>12}{'Repl.':>8}")
    print("-" * 70)
    for row in rows:
        print(f"{row['config']:<14}{row['best']:>12.2f}"
              f"{row['avg']:>12.2f}{row['worst']:>12.2f}"
              f"{row['avg_time']:>12.3f}{row['replicas']:>8d}")
    print(f"\nTiempo total de pared (wall clock): {t_total:.2f} s "
          f"con {jobs} proceso(s) paralelo(s).")
    print(f"\nMejor configuracion: {overall_best_cfg}")
    print_solution(overall_best_res, inst)
    return rows, overall_best_res, overall_best_cfg


# ===========================================================================
# 10. AUTO-PRUEBA DE VALIDACION
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
    assert feas and abs(val - manual2) < 1e-9, f"got {val}, exp {manual2}"
    inst2 = Instance(name="t2", demand=[0, 5, 5], Q=20, dist=dist,
                     service=[0, 7, 0])
    assert abs(trip_internal_latency([1, 2], inst2)
               - (10 + (10 + 7 + 12))) < 1e-9

    # Coherencia entre evaluate y evaluate_metrics
    sol3 = [[2, 1], [3]]
    va, _ = evaluate(sol3, inst)
    loads, durs, lats = [], [], []
    for t in sol3:
        L, D, La = compute_trip_metrics(t, inst)
        loads.append(L); durs.append(D); lats.append(La)
    vb, _ = evaluate_metrics(sol3, loads, durs, lats, inst)
    assert abs(va - vb) < 1e-9

    print("Auto-prueba de latencia (con y sin servicio): OK")


# ===========================================================================
# 11. PUNTO DE ENTRADA / LINEA DE COMANDOS
# ===========================================================================

DEFAULT_CONFIGS = [
    {"name": "C1-nn-a02",  "mode": "nn",  "alpha": 0.2, "max_iter": 80},
    {"name": "C2-nn-a04",  "mode": "nn",  "alpha": 0.4, "max_iter": 80},
    {"name": "C3-ins-a03", "mode": "ins", "alpha": 0.3, "max_iter": 80},
    {"name": "C4-nn-a00",  "mode": "nn",  "alpha": 0.0, "max_iter": 80},
]


def main():
    ap = argparse.ArgumentParser(
        description="GRASP para MTVRP con minima latencia (optimizado).")
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--round", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jobs", type=int, default=0,
                    help="Procesos paralelos (0 = todos los nucleos).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return
    _self_test()

    jobs = args.jobs if args.jobs > 0 else cpu_count()
    print(f"Usando {jobs} proceso(s) paralelo(s) "
          f"(detectados {cpu_count()} nucleos).")

    if not args.instances:
        print("Indica al menos un archivo de instancia. Ejemplo:")
        print("  python3 mtvrp_grasp_fast.py MT-DMP10s0-01.txt --jobs 8")
        return

    for path in args.instances:
        if not os.path.isfile(path):
            print(f"Aviso: no se encontro '{path}', se omite.")
            continue
        inst = read_instance(path, round_dist=args.round)
        print(inst)
        run_experiment(inst, DEFAULT_CONFIGS, replicas=args.replicas,
                        time_limit=args.time_limit, jobs=jobs)


if __name__ == "__main__":
    main()