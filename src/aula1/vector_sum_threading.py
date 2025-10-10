#!/usr/bin/env python3
"""
Exemplo 3: Soma de vetores com threading
Demonstra limitação do GIL para operações CPU-bound.
"""

import time
import numpy as np
import threading

def vector_sum_threading(a, b, num_threads=4):
    """Soma usando threading (limitado pelo GIL)"""
    start = time.perf_counter()
    
    chunk_size = len(a) // num_threads
    results = [None] * num_threads
    threads = []
    
    def worker(thread_id):
        start_idx = thread_id * chunk_size
        end_idx = start_idx + chunk_size if thread_id < num_threads - 1 else len(a)
        
        # Operação elemento por elemento (CPU-bound)
        chunk_result = np.zeros(end_idx - start_idx, dtype=a.dtype)
        for i in range(len(chunk_result)):
            chunk_result[i] = a[start_idx + i] + b[start_idx + i]
        
        results[thread_id] = chunk_result
    
    # Criar e iniciar threads
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Aguardar conclusão
    for thread in threads:
        thread.join()
    
    # Combinar resultados
    c = np.concatenate(results)
    end = time.perf_counter()
    return c, end - start

def main():
    print("🧵 Soma de Vetores com Threading")
    print("=" * 40)
    print("⚠️  Demonstra limitação do GIL para operações CPU-bound\n")
    
    N = 1_000_000
    print(f"Tamanho: {N:,} elementos")
    
    # Criar vetores
    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    
    # Teste com diferentes números de threads
    thread_counts = [1, 2, 4, 8]
    
    for num_threads in thread_counts:
        c, time_taken = vector_sum_threading(a, b, num_threads)
        
        print(f"{num_threads} threads: {time_taken:.4f}s")
        
        # Verificar resultado
        expected = a + b
        is_correct = np.allclose(c, expected)
        print(f"  Resultado correto: {'✓' if is_correct else '✗'}")
    
    print("\n💡 Observação: Threading não acelera operações CPU-bound")
    print("   devido ao Global Interpreter Lock (GIL) do Python")

if __name__ == "__main__":
    main()