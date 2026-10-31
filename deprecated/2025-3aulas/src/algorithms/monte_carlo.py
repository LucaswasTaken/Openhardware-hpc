"""
Monte Carlo Pi Estimation Algorithms
Educational examples showing serial vs threading vs multiprocessing
"""

import time
import numpy as np
import threading


def monte_carlo_pi_serial(n_samples, seed=42):
    """
    Estimativa de π serial usando Monte Carlo
    
    Args:
        n_samples: Número de amostras para simulação
        seed: Seed para reprodutibilidade
    
    Returns:
        tuple: (pi_estimate, execution_time)
    """
    start = time.perf_counter()
    
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Pontos dentro do círculo unitário
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    end = time.perf_counter()
    return pi_estimate, end - start


def monte_carlo_pi_threading(n_samples, n_threads=4, seed=42):
    """
    Estimativa de π usando threading (demonstração das limitações do GIL)
    
    Args:
        n_samples: Número de amostras para simulação
        n_threads: Número de threads
        seed: Seed base para reprodutibilidade
    
    Returns:
        tuple: (pi_estimate, execution_time)
    """
    start = time.perf_counter()
    
    samples_per_thread = n_samples // n_threads
    results = [0] * n_threads
    threads = []
    
    def worker(thread_id):
        # Usar seed diferente para cada thread
        local_state = np.random.RandomState(seed + thread_id)
        x = local_state.uniform(-1, 1, samples_per_thread)
        y = local_state.uniform(-1, 1, samples_per_thread)
        inside_circle = (x**2 + y**2) <= 1
        results[thread_id] = np.sum(inside_circle)
    
    # Criar e iniciar threads
    for i in range(n_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Aguardar conclusão
    for thread in threads:
        thread.join()
    
    total_inside = sum(results)
    pi_estimate = 4 * total_inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start


def run_monte_carlo_comparison(sample_sizes=None, n_threads=4):
    """
    Executa comparação entre métodos de Monte Carlo
    
    Args:
        sample_sizes: Lista de tamanhos de amostra para testar
        n_threads: Número de threads para teste
    
    Returns:
        dict: Resultados da comparação
    """
    if sample_sizes is None:
        sample_sizes = [20000, 1000000]
    
    results = {}
    
    print("🔍 Monte Carlo Pi Estimation: Threading vs Serial (GIL Demonstration)")
    print("=" * 70)
    
    for n_samples in sample_sizes:
        print(f"\n📊 Amostras: {n_samples:,}")
        
        # Serial
        pi_serial, time_serial = monte_carlo_pi_serial(n_samples)
        
        # Threading (shows GIL limitation)
        pi_threading, time_threading = monte_carlo_pi_threading(n_samples, n_threads)
        
        # Calculate speedup
        threading_speedup = time_serial / time_threading if time_threading > 0 else 1.0
        
        # Calculate errors
        error_serial = abs(pi_serial - np.pi)
        error_threading = abs(pi_threading - np.pi)
        
        print(f"  Serial:    π ≈ {pi_serial:.6f}, erro = {error_serial:.6f}, tempo = {time_serial:.4f}s")
        print(f"  Threading: π ≈ {pi_threading:.6f}, erro = {error_threading:.6f}, tempo = {time_threading:.4f}s")
        print(f"  Speedup:   {threading_speedup:.2f}x (limitado pelo GIL)")
        
        results[n_samples] = {
            'serial': {'pi': pi_serial, 'time': time_serial, 'error': error_serial},
            'threading': {'pi': pi_threading, 'time': time_threading, 'error': error_threading},
            'speedup': threading_speedup
        }
    
    print(f"\n📊 Valor real de π: {np.pi:.6f}")
    
    return results


def run_multiprocessing_demo(samples=100000000, processes=8):
    """
    Executa demonstração de multiprocessing real via script externo
    
    Args:
        samples: Número de amostras
        processes: Número de processos
    """
    import subprocess
    import sys
    import os
    
    print("🎯 Monte Carlo Pi Estimation (REAL MULTIPROCESSING)")
    print("-" * 50)
    
    # Path to multiprocessing script
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aula1_multiprocessing')
    monte_carlo_script = os.path.join(script_dir, 'monte_carlo_mp.py')
    
    if os.path.exists(monte_carlo_script):
        try:
            print(f"🔬 Testing with {samples:,} samples, {processes} processes:")
            
            result = subprocess.run([
                sys.executable, monte_carlo_script, 
                '--samples', str(samples),
                '--processes', str(processes)
            ], capture_output=True, text=True, cwd=script_dir, timeout=30)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ Monte Carlo script timed out")
        except Exception as e:
            print(f"❌ Failed to run Monte Carlo script: {e}")
    else:
        print(f"❌ Monte Carlo script not found: {monte_carlo_script}")