#!/usr/bin/env python3
"""
Exemplo 5: Multiplicação de matrizes por blocos com multiprocessing
Demonstra paralelização de operações matriciais.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def matrix_multiply_block_worker(args):
    """Worker que multiplica um bloco da matriz"""
    A_block, B, start_row, end_row = args
    return start_row, np.dot(A_block, B)

def matrix_multiply_parallel(A, B, n_processes):
    """Multiplicação de matrizes paralela por blocos de linhas"""
    start = time.perf_counter()
    
    rows_per_process = A.shape[0] // n_processes
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Dividir matriz A em blocos de linhas
        tasks = []
        for i in range(n_processes):
            start_row = i * rows_per_process
            end_row = start_row + rows_per_process if i < n_processes - 1 else A.shape[0]
            
            A_block = A[start_row:end_row, :]
            tasks.append((A_block, B, start_row, end_row))
        
        # Submeter tarefas
        futures = [executor.submit(matrix_multiply_block_worker, task) for task in tasks]
        
        # Coletar resultados
        results = [future.result() for future in as_completed(futures)]
    
    # Reconstituir matriz resultado
    results.sort(key=lambda x: x[0])  # Ordenar por start_row
    C = np.vstack([result[1] for result in results])
    
    end = time.perf_counter()
    return C, end - start

def matrix_multiply_serial(A, B):
    """Multiplicação serial usando NumPy"""
    start = time.perf_counter()
    C = np.dot(A, B)
    end = time.perf_counter()
    return C, end - start

def main():
    print("🧮 Multiplicação de Matrizes por Blocos")
    print("=" * 45)
    
    # Testar diferentes tamanhos
    matrix_sizes = [512, 1024, 2048]
    n_processes = mp.cpu_count()
    
    print(f"Usando {n_processes} processos\n")
    print("Tamanho    NumPy (s)    Paralelo (s)    Speedup    GFLOPS")
    print("-" * 60)
    
    for size in matrix_sizes:
        # Criar matrizes aleatórias
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Serial (NumPy)
        C_serial, time_serial = matrix_multiply_serial(A, B)
        
        # Paralelo
        C_parallel, time_parallel = matrix_multiply_parallel(A, B, n_processes)
        
        # Verificar correção
        are_equal = np.allclose(C_serial, C_parallel, rtol=1e-5)
        
        # Calcular métricas
        speedup = time_serial / time_parallel
        flops = 2 * size**3  # Multiplicação de matrizes: 2*n³ operações
        gflops_serial = flops / (time_serial * 1e9)
        gflops_parallel = flops / (time_parallel * 1e9)
        
        status = "✓" if are_equal else "✗"
        print(f"{size:>6}x{size:<6} {time_serial:>8.3f}    {time_parallel:>10.3f}    {speedup:>6.2f}x   {gflops_parallel:>6.2f} {status}")
    
    print("\n💡 Observações:")
    print("• NumPy já usa BLAS otimizado (pode ser paralelo internamente)")
    print("• Speedup depende da implementação BLAS do sistema")
    print("• Para ganhos reais, considere bibliotecas especializadas")

if __name__ == "__main__":
    main()