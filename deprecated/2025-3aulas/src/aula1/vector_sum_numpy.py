#!/usr/bin/env python3
"""
Exemplo 2: Soma de vetores com NumPy
Demonstra paralelismo implícito via operações vetorizadas.
"""

import time
import numpy as np

def vector_sum_numpy(a, b):
    """Soma usando NumPy (paralelismo implícito)"""
    start = time.perf_counter()
    c = a + b
    end = time.perf_counter()
    return c, end - start

def main():
    print("⚡ Soma de Vetores com NumPy")
    print("=" * 35)
    
    # Teste com diferentes tamanhos
    sizes = [1_000_000, 10_000_000, 50_000_000]
    
    for N in sizes:
        print(f"\nTamanho: {N:,} elementos")
        
        # Criar vetores
        a = np.arange(N, dtype=np.float64)
        b = np.arange(N, dtype=np.float64)
        
        # Soma NumPy
        c, time_taken = vector_sum_numpy(a, b)
        
        # Calcular throughput
        throughput = N / time_taken / 1e6  # Melementos/s
        
        print(f"Tempo NumPy: {time_taken:.4f}s")
        print(f"Throughput: {throughput:.1f} Melementos/s")
        print(f"Resultado[0:5]: {c[:5]}")

if __name__ == "__main__":
    main()