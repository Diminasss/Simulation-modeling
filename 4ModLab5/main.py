#!/usr/bin/env python3
"""
loss_mo.py

Simulate a single-server loss system (no queue, no priorities).
- Arrival process: poisson (exponential interarrival) or erlang (Erlang-k interarrival)
- Service distribution: exponential, erlang, deterministic, gamma
- No plotting — prints statistics and can save CSV files with departures and inter-departure times.

Usage:
    python loss_mo.py --help
"""

import argparse
import math
import csv
from typing import List
import numpy as np


# ---------------- Random generators (depend on numpy) ----------------
def gen_interarrival_poisson(rate: float) -> float:
    return np.random.exponential(1.0 / rate)


def gen_interarrival_erlang(rate: float, k: int) -> float:
    # Erlang-k with mean = 1/rate -> Gamma(shape=k, scale=1/(rate*k))
    return np.random.gamma(k, 1.0 / (rate * k))


def gen_service_exponential(mean: float) -> float:
    return np.random.exponential(mean)


def gen_service_erlang(mean: float, k: int) -> float:
    # Erlang with mean = mean -> gamma(shape=k, scale=mean/k)
    return np.random.gamma(k, mean / k)


def gen_service_deterministic(mean: float) -> float:
    return float(mean)


def gen_service_gamma(mean: float, shape: float) -> float:
    # gamma with given shape; scale such that mean == shape*scale
    scale = mean / shape
    return np.random.gamma(shape, scale)


# ---------------- Simulation core ----------------
def simulate_loss_system(
        arrival_process: str,
        arrival_rate: float,
        erlang_k_arrival: int,
        service_dist: str,
        service_mean: float,
        erlang_k_service: int,
        gamma_shape: float,
        max_departures: int,
        rng_seed: int = None,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    departures: List[float] = []
    arrivals_total = 0
    accepted = 0
    lost = 0

    t = 0.0
    server_busy = False
    next_departure = math.inf

    # schedule first arrival
    if arrival_process == 'poisson':
        next_arrival = t + gen_interarrival_poisson(arrival_rate)
    elif arrival_process == 'erlang':
        next_arrival = t + gen_interarrival_erlang(arrival_rate, erlang_k_arrival)
    else:
        raise ValueError("Unknown arrival_process")

    def gen_service():
        if service_dist == 'exponential':
            return gen_service_exponential(service_mean)
        elif service_dist == 'erlang':
            return gen_service_erlang(service_mean, erlang_k_service)
        elif service_dist == 'deterministic':
            return gen_service_deterministic(service_mean)
        elif service_dist == 'gamma':
            return gen_service_gamma(service_mean, gamma_shape)
        else:
            raise ValueError("Unknown service_dist")

    # main event loop
    while len(departures) < max_departures:
        if next_arrival <= next_departure:
            t = next_arrival
            arrivals_total += 1

            if not server_busy:
                accepted += 1
                s = gen_service()
                next_departure = t + s
                server_busy = True
            else:
                lost += 1

            # schedule next arrival
            if arrival_process == 'poisson':
                next_arrival = t + gen_interarrival_poisson(arrival_rate)
            else:
                next_arrival = t + gen_interarrival_erlang(arrival_rate, erlang_k_arrival)
        else:
            # departure
            t = next_departure
            departures.append(t)
            server_busy = False
            next_departure = math.inf

    departures_arr = np.array(departures)
    inter_departures = np.diff(departures_arr) if departures_arr.size > 1 else np.array([])

    # statistics
    stats = {}
    stats['arrivals_total'] = arrivals_total
    stats['accepted'] = accepted
    stats['lost'] = lost
    stats['departures_collected'] = len(departures_arr)

    if inter_departures.size > 0:
        mean_id = float(np.mean(inter_departures))
        var_id = float(np.var(inter_departures, ddof=1))
        cv = float(math.sqrt(var_id) / mean_id) if mean_id != 0 else float('nan')
        throughput = 1.0 / mean_id if mean_id > 0 else float('nan')

        stats.update({
            'mean_inter_departure': mean_id,
            'var_inter_departure': var_id,
            'cv_inter_departure': cv,
            'throughput_empirical': throughput,
        })
    else:
        stats.update({
            'mean_inter_departure': None,
            'var_inter_departure': None,
            'cv_inter_departure': None,
            'throughput_empirical': None,
        })

    meta = {
        'arrival_process': arrival_process,
        'arrival_rate': arrival_rate,
        'erlang_k_arrival': erlang_k_arrival,
        'service_dist': service_dist,
        'service_mean': service_mean,
        'erlang_k_service': erlang_k_service,
        'gamma_shape': gamma_shape,
    }

    return departures_arr, inter_departures, stats, meta


# ---------------- CSV save helpers ----------------
def save_csv_times(filename: str, times: np.ndarray):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'time'])
        for i, t in enumerate(times, start=1):
            writer.writerow([i, f"{t:.12f}"])


def save_csv_intervals(filename: str, intervals: np.ndarray):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'interval'])
        for i, dt in enumerate(intervals, start=1):
            writer.writerow([i, f"{dt:.12f}"])


# ---------------- CLI and main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Simulate single-server loss system (no queue).")
    p.add_argument('--arrival-process', choices=['poisson', 'erlang'], default='poisson',
                   help='Type of interarrival: poisson (exponential) or erlang')
    p.add_argument('--arrival-rate', type=float, default=1.0, help='Arrival rate (lambda), arrivals per unit time')
    p.add_argument('--erlang-k-arrival', type=int, default=3, help='k for Erlang arrival (if erlang chosen)')
    p.add_argument('--service-dist', choices=['exponential', 'erlang', 'deterministic', 'gamma'],
                   default='exponential', help='Service time distribution')
    p.add_argument('--service-mean', type=float, default=0.8, help='Mean service time')
    p.add_argument('--erlang-k-service', type=int, default=2, help='k for Erlang service (if erlang chosen)')
    p.add_argument('--gamma-shape', type=float, default=2.0, help='Shape parameter for Gamma service')
    p.add_argument('--max-departures', type=int, default=3000, help='How many departures to collect')
    p.add_argument('--seed', type=int, default=12345, help='RNG seed (integer) or omit for random')
    p.add_argument('--save-departures', type=str, default='', help='Filename to save departure times CSV (optional)')
    p.add_argument('--save-intervals', type=str, default='',
                   help='Filename to save inter-departure intervals CSV (optional)')
    return p.parse_args()


def main():
    args = parse_args()

    rng_seed = args.seed if args.seed is not None else None

    deps, intervals, stats, meta = simulate_loss_system(
        arrival_process=args.arrival_process,
        arrival_rate=args.arrival_rate,
        erlang_k_arrival=args.erlang_k_arrival,
        service_dist=args.service_dist,
        service_mean=args.service_mean,
        erlang_k_service=args.erlang_k_service,
        gamma_shape=args.gamma_shape,
        max_departures=args.max_departures,
        rng_seed=rng_seed,
    )

    # Print summary
    print("Simulation summary")
    print("==================")
    print(
        f"Arrival process: {meta['arrival_process']} (lambda={meta['arrival_rate']}, erlang_k_arrival={meta['erlang_k_arrival']})")
    print(
        f"Service dist: {meta['service_dist']} (mean={meta['service_mean']}, erlang_k_service={meta['erlang_k_service']}, gamma_shape={meta['gamma_shape']})")
    print()
    print(f"Total arrivals generated (during sim): {stats['arrivals_total']}")
    print(f"Accepted (served): {stats['accepted']}")
    print(f"Lost (blocked): {stats['lost']}")
    print(f"Collected departures: {stats['departures_collected']}")
    print()

    if stats['mean_inter_departure'] is not None:
        print("Inter-departure statistics")
        print("-------------------------")
        print(f"Mean inter-departure time: {stats['mean_inter_departure']:.12f}")
        print(f"Variance: {stats['var_inter_departure']:.12f}")
        print(f"Coefficient of variation (CV): {stats['cv_inter_departure']:.6f}")
        print(f"Throughput (empirical departures per time unit): {stats['throughput_empirical']:.12f}")
        print()
        print("Simple heuristic:")
        print("- If inter-departure times are exponential, CV ≈ 1 (exp CV = 1).")
        print(
            "- For M/M/1 queue with infinite buffer (not this loss system), Burke's theorem gives Poisson departures.")
        print("- In this loss system (no queue), output may differ; use CV and saved intervals for deeper tests.")
    else:
        print("Too few departures to compute inter-departure statistics.")

    # Save CSVs if requested
    if args.save_departures:
        save_csv_times(args.save_departures, deps)
        print(f"Departure times saved to: {args.save_departures}")
    if args.save_intervals:
        save_csv_intervals(args.save_intervals, intervals)
        print(f"Inter-departure intervals saved to: {args.save_intervals}")


if __name__ == '__main__':
    main()
