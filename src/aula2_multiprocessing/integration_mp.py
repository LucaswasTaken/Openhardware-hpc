#!/usr/bin/env python3
"""
Real Multiprocessing Integration - Aula 2
Integração numérica paralela usando ProcessPoolExecutor
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def function_to_integrate(x):
    """Função exemplo: f(x) = x²·sin(x) + cos(x)"""
    return x**2 * np.sin(x) + np.cos(x)


def trapezoidal_rule_chunk(args):
    """Integração trapezoidal para um chunk do domínio"""
    start, end, n_points = args
    
    x = np.linspace(start, end, n_points)
    y = function_to_integrate(x)
    
    # Regra do trapézio
    h = (end - start) / (n_points - 1)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    return integral


def integrate_parallel(a, b, n_total, n_processes):
    """Integração paralela usando futures"""
    start_time = time.perf_counter()
    
    # Dividir domínio entre processos
    chunk_width = (b - a) / n_processes
    points_per_chunk = n_total // n_processes
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = []
        
        for i in range(n_processes):
            chunk_start = a + i * chunk_width
            chunk_end = chunk_start + chunk_width
            
            # Último chunk pode ter pontos extras
            if i == n_processes - 1:
                chunk_end = b
                points_per_chunk = n_total - i * points_per_chunk
            
            args = (chunk_start, chunk_end, points_per_chunk)
            future = executor.submit(trapezoidal_rule_chunk, args)
            futures.append(future)
        
        # Coletar resultados
        total_integral = sum(future.result() for future in as_completed(futures))
    
    end_time = time.perf_counter()
    return total_integral, end_time - start_time


def integrate_serial(a, b, n_points):
    """Integração serial para comparação"""
    start_time = time.perf_counter()
    
    x = np.linspace(a, b, n_points)
    y = function_to_integrate(x)
    
    h = (b - a) / (n_points - 1)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    end_time = time.perf_counter()
    return integral, end_time - start_time


def run_integration_comparison(n_points, process_counts):
    """Executa comparação de integração com diferentes números de processos"""
    
    # Parâmetros de integração
    a, b = 0, 10  # Domínio de integração
    
    print(f"🧮 Integração Numérica f(x) = x²·sin(x) + cos(x)")
    print(f"Domínio: [{a}, {b}], Pontos: {n_points:,}")
    print("=" * 60)
    
    results = []
    
    # Testar diferentes números de processos
    for n_proc in process_counts:
        if n_proc == 1:
            # Usar versão serial
            integral, exec_time = integrate_serial(a, b, n_points)
            method = "Serial"
        else:
            # Usar versão paralela
            integral, exec_time = integrate_parallel(a, b, n_points, n_proc)
            method = f"{n_proc} proc"
        
        results.append({
            'processes': n_proc,
            'integral': integral,
            'time': exec_time,
            'method': method
        })
        
        print(f"{method:>10}: Integral = {integral:.6f}, Tempo = {exec_time:.3f}s")
    
    # Calcular speedups
    serial_time = results[0]['time']
    print(f"\n📊 Análise de Speedup:")
    
    for result in results[1:]:
        speedup = serial_time / result['time']
        efficiency = speedup / result['processes']
        print(f"  {result['processes']} processos: Speedup = {speedup:.2f}x, Eficiência = {efficiency:.2f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integração numérica com multiprocessing')
    parser.add_argument('--points', type=int, default=5000000, help='Número de pontos de integração')
    parser.add_argument('--processes', type=str, default='1,2,4', 
                        help='Comma-separated list of process counts to test (default: 1,2,4)')
    
    args = parser.parse_args()
    
    # Parse process counts
    try:
        process_counts = [int(p.strip()) for p in args.processes.split(',')]
    except ValueError:
        print("❌ Erro: processes deve ser uma lista separada por vírgulas (ex: 1,2,4)")
        exit(1)
    
    results = run_integration_comparison(args.points, process_counts)