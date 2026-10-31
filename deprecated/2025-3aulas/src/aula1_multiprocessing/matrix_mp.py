#!/usr/bin/env python3
"""
Real Multiprocessing Matrix Operations
This script demonstrates actual multiprocessing for matrix computations.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse

def matrix_chunk_worker(args):
    """Worker function for matrix multiplication by rows"""
    A_chunk, B, start_row = args
    return start_row, np.dot(A_chunk, B)

def matrix_multiply_multiprocessing(A, B, n_processes=None):
    """Real multiprocessing matrix multiplication"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    start = time.perf_counter()
    
    rows_per_process = A.shape[0] // n_processes
    
    # Prepare chunks for each process
    args_list = []
    for i in range(n_processes):
        start_row = i * rows_per_process
        end_row = start_row + rows_per_process if i < n_processes - 1 else A.shape[0]
        A_chunk = A[start_row:end_row, :]
        args_list.append((A_chunk, B, start_row))
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(matrix_chunk_worker, args_list))
    
    # Sort results by position and combine
    results.sort(key=lambda x: x[0])
    C = np.vstack([result[1] for result in results])
    
    end = time.perf_counter()
    return C, end - start

def custom_algorithm_worker(args):
    """Worker for custom algorithm that benefits from multiprocessing"""
    A_chunk, B, operation_type = args
    
    if operation_type == "element_wise_power":
        # Custom operation: (A @ B)^2 + sin(A @ B)
        result = np.dot(A_chunk, B)
        result = result**2 + np.sin(result)
        return result
    elif operation_type == "iterative_refinement":
        # Simulation of iterative refinement
        result = np.dot(A_chunk, B)
        for _ in range(10):
            result = result + 0.01 * np.sin(result)
        return result

def custom_algorithm_multiprocessing(A, B, operation_type, n_processes=None):
    """Custom algorithm using real multiprocessing"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    start = time.perf_counter()
    
    rows_per_process = A.shape[0] // n_processes
    
    # Prepare chunks
    args_list = []
    for i in range(n_processes):
        start_row = i * rows_per_process
        end_row = start_row + rows_per_process if i < n_processes - 1 else A.shape[0]
        A_chunk = A[start_row:end_row, :]
        args_list.append((A_chunk, B, operation_type))
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(custom_algorithm_worker, args_list))
    
    C = np.vstack(results)
    
    end = time.perf_counter()
    return C, end - start

def custom_algorithm_serial(A, B, operation_type):
    """Serial version of custom algorithm"""
    start = time.perf_counter()
    
    if operation_type == "element_wise_power":
        result = np.dot(A, B)
        result = result**2 + np.sin(result)
    elif operation_type == "iterative_refinement":
        result = np.dot(A, B)
        for _ in range(10):
            result = result + 0.01 * np.sin(result)
    
    end = time.perf_counter()
    return result, end - start

def matrix_multiply_serial(A, B):
    """Serial matrix multiplication using NumPy"""
    start = time.perf_counter()
    C = np.dot(A, B)
    end = time.perf_counter()
    return C, end - start

def run_matrix_comparison(size, n_processes=None):
    """Run comparison of matrix operations"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"🔢 Matrix Operations - REAL MULTIPROCESSING")
    print(f"Matrix size: {size}x{size}")
    print(f"Processes: {n_processes}")
    print(f"CPU cores: {mp.cpu_count()}")
    print("=" * 60)
    
    # Create matrices
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Test standard multiplication
    C_serial, time_serial = matrix_multiply_serial(A, B)
    C_mp, time_mp = matrix_multiply_multiprocessing(A, B, n_processes)
    
    # Verify results are equal
    mp_correct = np.allclose(C_serial, C_mp, rtol=1e-5)
    mp_speedup = time_serial / time_mp if time_mp > 0 else 1.0
    mp_efficiency = mp_speedup / n_processes
    
    print(f"\nStandard Matrix Multiplication:")
    print(f"  NumPy (BLAS):     {time_serial:.4f}s")
    print(f"  Multiprocessing:  {time_mp:.4f}s {'✓' if mp_correct else '✗'}")
    print(f"  Speedup:          {mp_speedup:.2f}x")
    print(f"  Efficiency:       {mp_efficiency:.2f} ({mp_efficiency*100:.1f}%)")
    
    # Test custom algorithm where multiprocessing should shine
    print(f"\nCustom Algorithm (element_wise_power):")
    operation = "element_wise_power"
    
    C_custom_serial, time_custom_serial = custom_algorithm_serial(A, B, operation)
    C_custom_mp, time_custom_mp = custom_algorithm_multiprocessing(A, B, operation, n_processes)
    
    custom_correct = np.allclose(C_custom_serial, C_custom_mp, rtol=1e-4)
    custom_speedup = time_custom_serial / time_custom_mp if time_custom_mp > 0 else 1.0
    custom_efficiency = custom_speedup / n_processes
    
    print(f"  Serial:           {time_custom_serial:.4f}s")
    print(f"  Multiprocessing:  {time_custom_mp:.4f}s {'✓' if custom_correct else '✗'}")
    print(f"  Speedup:          {custom_speedup:.2f}x")
    print(f"  Efficiency:       {custom_efficiency:.2f} ({custom_efficiency*100:.1f}%)")
    
    return {
        'size': size,
        'standard': {
            'time_serial': time_serial,
            'time_mp': time_mp,
            'speedup': mp_speedup,
            'efficiency': mp_efficiency,
            'correct': mp_correct
        },
        'custom': {
            'time_serial': time_custom_serial,
            'time_mp': time_custom_mp,
            'speedup': custom_speedup,
            'efficiency': custom_efficiency,
            'correct': custom_correct
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matrix operations with multiprocessing')
    parser.add_argument('--size', type=int, default=500, help='Matrix size')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes')
    
    args = parser.parse_args()
    
    results = run_matrix_comparison(args.size, args.processes)