#!/usr/bin/env python3
"""
Real Multiprocessing Monte Carlo Pi Estimation
This script demonstrates actual multiprocessing for Monte Carlo simulation.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import sys
import argparse

def monte_carlo_worker(args):
    """Worker function for multiprocessing - calculates pi for a chunk of samples"""
    n_samples, process_id = args
    
    # Use unique seed for each process
    np.random.seed(42 + process_id * 1000)
    
    # Generate random points
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Count points inside the unit circle
    inside_circle = (x**2 + y**2) <= 1
    return np.sum(inside_circle)

def monte_carlo_pi_multiprocessing(n_samples, n_processes=None):
    """Real multiprocessing Monte Carlo pi estimation"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    start = time.perf_counter()
    
    samples_per_process = n_samples // n_processes
    
    # Prepare arguments for each process
    args_list = [(samples_per_process, i) for i in range(n_processes)]
    
    # Execute in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(monte_carlo_worker, args_list))
    
    total_inside = sum(results)
    pi_estimate = 4 * total_inside / n_samples
    
    end = time.perf_counter()
    return pi_estimate, end - start, results

def monte_carlo_pi_serial(n_samples):
    """Serial Monte Carlo pi estimation for comparison"""
    start = time.perf_counter()
    
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    end = time.perf_counter()
    return pi_estimate, end - start

def run_comparison(n_samples, n_processes=None):
    """Run comparison between serial and multiprocessing versions"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"🎯 Monte Carlo Pi Estimation - REAL MULTIPROCESSING")
    print(f"Samples: {n_samples:,}")
    print(f"Processes: {n_processes}")
    print(f"CPU cores: {mp.cpu_count()}")
    print("=" * 60)
    
    # Serial version
    pi_serial, time_serial = monte_carlo_pi_serial(n_samples)
    
    # Multiprocessing version
    pi_mp, time_mp, process_results = monte_carlo_pi_multiprocessing(n_samples, n_processes)
    
    # Calculate metrics
    speedup = time_serial / time_mp if time_mp > 0 else 1.0
    efficiency = speedup / n_processes
    error_serial = abs(pi_serial - np.pi)
    error_mp = abs(pi_mp - np.pi)
    
    # Results
    print(f"\nResults:")
    print(f"  Serial:        π ≈ {pi_serial:.6f}, error = {error_serial:.6f}, time = {time_serial:.4f}s")
    print(f"  Multiprocess:  π ≈ {pi_mp:.6f}, error = {error_mp:.6f}, time = {time_mp:.4f}s")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"  Efficiency:    {efficiency:.2f} ({efficiency*100:.1f}%)")
    print(f"  True π:        {np.pi:.6f}")
    
    print(f"\nWork distribution per process:")
    for i, count in enumerate(process_results):
        print(f"  Process {i}: {count:,} points inside circle")
    
    return {
        'pi_serial': pi_serial,
        'time_serial': time_serial,
        'pi_mp': pi_mp,
        'time_mp': time_mp,
        'speedup': speedup,
        'efficiency': efficiency,
        'process_results': process_results
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monte Carlo Pi estimation with multiprocessing')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of samples')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes')
    
    args = parser.parse_args()
    
    results = run_comparison(args.samples, args.processes)