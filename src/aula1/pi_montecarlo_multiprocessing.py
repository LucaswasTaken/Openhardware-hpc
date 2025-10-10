#!/usr/bin/env python3
"""
Exemplo 4: Estimativa de π usando Monte Carlo com multiprocessing
Demonstra paralelismo real através de processos separados.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def monte_carlo_worker(n_samples_per_process):
    """Worker function para cada processo"""
    np.random.seed()  # Seed diferente para cada processo
    x = np.random.uniform(-1, 1, n_samples_per_process)
    y = np.random.uniform(-1, 1, n_samples_per_process)
    
    inside_circle = (x**2 + y**2) <= 1
    return np.sum(inside_circle)

def monte_carlo_pi_parallel(n_samples, n_processes):
    """Estimativa de π paralela usando multiprocessing"""
    start = time.perf_counter()
    
    samples_per_process = n_samples // n_processes
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submeter tarefas para todos os processos
        futures = [executor.submit(monte_carlo_worker, samples_per_process) 
                  for _ in range(n_processes)]
        
        # Coletar resultados
        total_inside = sum(future.result() for future in as_completed(futures))
    
    pi_estimate = 4 * total_inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start

def monte_carlo_pi_serial(n_samples):
    """Estimativa de π serial para comparação"""
    start = time.perf_counter()
    
    np.random.seed(42)  # Para reprodutibilidade
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    end = time.perf_counter()
    return pi_estimate, end - start

def main():
    print("🎲 Estimativa de π usando Monte Carlo")
    print("=" * 45)
    
    n_processes = mp.cpu_count()
    print(f"Usando {n_processes} processos\n")
    
    # Teste com diferentes números de amostras
    sample_sizes = [1_000_000, 10_000_000, 50_000_000]
    
    print("Amostras       π (Serial)    Tempo (s)   π (Paralelo)  Tempo (s)   Speedup")
    print("-" * 75)
    
    for n_samples in sample_sizes:
        # Serial
        pi_serial, time_serial = monte_carlo_pi_serial(n_samples)
        
        # Paralelo
        pi_parallel, time_parallel = monte_carlo_pi_parallel(n_samples, n_processes)
        
        speedup = time_serial / time_parallel
        
        print(f"{n_samples:>9,}   {pi_serial:>8.6f}   {time_serial:>7.3f}   {pi_parallel:>8.6f}   {time_parallel:>7.3f}   {speedup:>6.2f}x")
    
    # Erro em relação ao valor real
    print(f"\n📊 Valor real de π: {np.pi:.6f}")
    print(f"💡 Monte Carlo é 'embaraçosamente paralelo' - ideal para multiprocessing")

if __name__ == "__main__":
    main()