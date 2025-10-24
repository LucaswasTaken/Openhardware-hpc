"""
Matrix Operations Algorithms
Educational examples showing serial vs threading vs multiprocessing
"""

import time
import numpy as np
import threading


def matrix_multiply_serial(A, B):
    """
    Multiplicação serial usando NumPy
    
    Args:
        A, B: Matrizes para multiplicação
    
    Returns:
        tuple: (result_matrix, execution_time)
    """
    start = time.perf_counter()
    C = np.dot(A, B)
    end = time.perf_counter()
    return C, end - start


def matrix_multiply_threading(A, B, n_threads=4):
    """
    Multiplicação usando threading para demonstração
    
    Args:
        A, B: Matrizes para multiplicação  
        n_threads: Número de threads
    
    Returns:
        tuple: (result_matrix, execution_time)
    """
    start = time.perf_counter()
    
    rows_per_thread = A.shape[0] // n_threads
    results = [None] * n_threads
    threads = []
    
    def worker(thread_id):
        start_row = thread_id * rows_per_thread
        end_row = start_row + rows_per_thread if thread_id < n_threads - 1 else A.shape[0]
        results[thread_id] = np.dot(A[start_row:end_row, :], B)
    
    # Criar e iniciar threads
    for i in range(n_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Aguardar conclusão
    for thread in threads:
        thread.join()
    
    # Combinar resultados
    C = np.vstack([r for r in results if r is not None])
    end = time.perf_counter()
    return C, end - start


def run_matrix_comparison(matrix_sizes=None, n_threads=4):
    """
    Executa comparação entre métodos de multiplicação de matriz
    
    Args:
        matrix_sizes: Lista de tamanhos de matriz para testar
        n_threads: Número de threads para teste
    
    Returns:
        dict: Resultados da comparação
    """
    if matrix_sizes is None:
        matrix_sizes = [500, 1000]
    
    results = {}
    
    print("🔍 Matrix Operations: Threading vs NumPy (GIL Impact Demonstration)")  
    print("=" * 70)
    
    for size in matrix_sizes:
        print(f"\n📊 Matrizes {size}x{size}")
        
        # Create matrices
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Test NumPy (baseline optimized)
        C_serial, time_serial = matrix_multiply_serial(A, B)
        
        # Test threading (shows overhead)
        C_threading, time_threading = matrix_multiply_threading(A, B, n_threads)
        
        # Verify results are equal
        threading_equal = np.allclose(C_serial, C_threading, rtol=1e-5)
        threading_speedup = time_serial / time_threading if time_threading > 0 else 1.0
        
        print(f"  NumPy (BLAS):     {time_serial:.4f}s")
        print(f"  Threading:        {time_threading:.4f}s, speedup: {threading_speedup:.2f}x {'✓' if threading_equal else '✗'}")
        
        if threading_speedup < 1:
            print(f"    ⚠️  Threading é mais lento devido ao overhead")
        elif threading_speedup > 1.2:
            print(f"    ⚡  Speedup inesperado - otimizações do sistema")
        else:
            print(f"    ⚠️  Speedup mínimo - NumPy já é otimizado com BLAS")
        
        results[size] = {
            'serial': {'time': time_serial},
            'threading': {'time': time_threading, 'equal': threading_equal},
            'speedup': threading_speedup
        }
    
    return results


def run_matrix_multiprocessing_demo(size=1000, processes=4):
    """
    Executa demonstração de multiprocessing real via script externo
    
    Args:
        size: Tamanho da matriz (size x size)
        processes: Número de processos
    """
    import subprocess
    import sys
    import os
    
    print("🔢 Matrix Operations (REAL MULTIPROCESSING)")
    print("-" * 50)
    
    # Path to multiprocessing script
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aula1_multiprocessing')
    matrix_script = os.path.join(script_dir, 'matrix_mp.py')
    
    if os.path.exists(matrix_script):
        try:
            print(f"🔬 Testing {size}x{size} matrices with {processes} processes:")
            
            result = subprocess.run([
                sys.executable, matrix_script,
                '--size', str(size),
                '--processes', str(processes)
            ], capture_output=True, text=True, cwd=script_dir, timeout=30)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ Matrix script timed out - use smaller matrices for quick demos")
        except Exception as e:
            print(f"❌ Failed to run script: {e}")
    else:
        print(f"❌ Script not found: {matrix_script}")