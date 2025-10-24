"""
Vector Operations - Demonstrating Serial vs NumPy vs Threading Performance

This module provides different implementations of vector operations to demonstrate
the limitations of threading in Python (GIL) and the efficiency of NumPy's
vectorized operations.
"""

import time
import numpy as np
import threading


def vector_sum_serial(a, b):
    """Soma serial elemento por elemento"""
    start = time.perf_counter()
    c = np.zeros_like(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    end = time.perf_counter()
    return c, end - start


def vector_sum_numpy(a, b):
    """Soma usando NumPy (paralelismo implícito)"""
    start = time.perf_counter()
    c = a + b
    end = time.perf_counter()
    return c, end - start


def vector_sum_threading(a, b, num_threads=4):
    """Soma usando threading (limitado pelo GIL)"""
    start = time.perf_counter()
    
    chunk_size = len(a) // num_threads
    threads = []
    results = [None] * num_threads
    
    def worker(thread_id):
        start_idx = thread_id * chunk_size
        end_idx = start_idx + chunk_size if thread_id < num_threads - 1 else len(a)
        
        for i in range(start_idx, end_idx):
            results[thread_id] = (a[start_idx:end_idx] + b[start_idx:end_idx])
    
    # Criar e iniciar threads
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Aguardar conclusão
    for thread in threads:
        thread.join()
    
    # Combinar resultados
    c = np.concatenate([r for r in results if r is not None])
    end = time.perf_counter()
    return c, end - start


def run_vector_sum_comparison(vector_size=500000, num_threads=4):
    """
    Execute comprehensive comparison of vector sum methods
    
    Args:
        vector_size: Size of vectors to create and sum
        num_threads: Number of threads for threading comparison
        
    Returns:
        dict: Results containing timing data and analysis
    """
    print(f"🧮 Comparação de Métodos de Soma de Vetores")
    print("=" * 50)
    print(f"Tamanho dos vetores: {vector_size:,} elementos")
    
    # Create test vectors
    print(f"Criando vetores de tamanho {vector_size:,}...")
    a = np.arange(vector_size, dtype=np.float64)
    b = np.arange(vector_size, dtype=np.float64)
    
    results = {}
    
    # Test serial method (only for reasonable sizes)
    if vector_size <= 1_000_000:
        print("\n📏 Executando soma serial...")
        _, time_serial = vector_sum_serial(a, b)
        results['serial'] = time_serial
        print(f"Serial:    {time_serial:.4f}s")
    else:
        print("\n⚠️  Serial pulado (vetor muito grande)")
        results['serial'] = None
    
    # Test NumPy method
    print("🚀 Executando soma NumPy...")
    _, time_numpy = vector_sum_numpy(a, b)
    results['numpy'] = time_numpy
    print(f"NumPy:     {time_numpy:.4f}s")
    
    # Test threading method  
    print(f"🧵 Executando soma Threading ({num_threads} threads)...")
    _, time_threading = vector_sum_threading(a, b, num_threads)
    results['threading'] = time_threading
    print(f"Threading: {time_threading:.4f}s")
    
    # Calculate and display speedups
    print("\n📊 Análise de Performance:")
    print("-" * 30)
    
    if results['serial']:
        speedup_numpy = results['serial'] / results['numpy']
        speedup_threading = results['serial'] / results['threading']
        print(f"Speedup NumPy vs Serial:    {speedup_numpy:.2f}x")
        print(f"Speedup Threading vs Serial: {speedup_threading:.2f}x")
    
    numpy_vs_threading = results['threading'] / results['numpy']
    print(f"Threading vs NumPy:         {numpy_vs_threading:.2f}x")
    
    if numpy_vs_threading > 1:
        print("✅ NumPy é mais rápido que Threading")
    else:
        print("⚠️  Threading superou NumPy (inesperado)")
    
    # Educational notes
    print("\n💡 Observações Importantes:")
    print("• NumPy usa bibliotecas otimizadas (BLAS) com paralelismo implícito")
    print("• Threading em Python é limitado pelo GIL para operações CPU-bound")
    print("• Para soma de vetores, NumPy é quase sempre a melhor opção")
    print("• Threading pode ter overhead que supera os benefícios")
    
    return results


def demonstrate_gil_limitation(n_iterations=50000000, n_threads=4):
    """
    Demonstra a limitação do GIL com trabalho CPU-intensivo puro
    
    Args:
        n_iterations: Número de iterações para o trabalho CPU
        n_threads: Número de threads para comparação
    """
    print(f"\n🔒 Demonstração da Limitação do GIL")
    print("=" * 45)
    print(f"Calculando soma de quadrados para {n_iterations:,} números...")
    
    def cpu_work_serial(n_iterations):
        """Trabalho CPU-intensivo serial"""
        start = time.perf_counter()
        total = 0
        for i in range(n_iterations):
            total += i ** 2
        end = time.perf_counter()
        return total, end - start

    def cpu_work_threading(n_iterations, n_threads=4):
        """Trabalho CPU-intensivo com threading"""
        start = time.perf_counter()
        
        work_per_thread = n_iterations // n_threads
        results = [0] * n_threads
        threads = []
        
        def worker(thread_id):
            start_range = thread_id * work_per_thread
            end_range = start_range + work_per_thread
            total = 0
            for i in range(start_range, end_range):
                total += i ** 2
            results[thread_id] = total
        
        # Criar e iniciar threads
        for i in range(n_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Aguardar conclusão
        for thread in threads:
            thread.join()
        
        total = sum(results)
        end = time.perf_counter()
        return total, end - start
    
    # Execute tests
    result_serial, time_serial = cpu_work_serial(n_iterations)
    result_threading, time_threading = cpu_work_threading(n_iterations, n_threads)
    
    # Calculate results
    results_match = (result_serial == result_threading)
    speedup = time_serial / time_threading
    
    print(f"\nResultados:")
    print(f"  Serial:    {time_serial:.4f}s")
    print(f"  Threading: {time_threading:.4f}s")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"  Precisão:  {'✓' if results_match else '✗'}")
    
    if speedup < 1.2:
        print("\n🔍 Como esperado: Threading não oferece speedup significativo")
        print("   Motivo: GIL impede paralelismo real em tarefas CPU-bound")
    else:
        print("\n🔍 Speedup inesperado - pode ser devido a otimizações do sistema")
    
    print(f"\n💡 Para paralelismo real em Python:")
    print("   • Use multiprocessing.Pool")
    print("   • Use joblib.Parallel") 
    print("   • Use numba com paralelização")
    print("   • Use bibliotecas como numpy que liberam o GIL internamente")
    
    return {
        'serial_time': time_serial,
        'threading_time': time_threading,
        'speedup': speedup,
        'results_match': results_match
    }