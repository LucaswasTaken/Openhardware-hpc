#!/usr/bin/env python3
"""
Exemplo 1: Soma de vetores serial
Demonstra operação básica elemento por elemento em Python puro.
"""

import time
import numpy as np

def vector_sum_serial(a, b):
    """Soma serial elemento por elemento"""
    start = time.perf_counter()
    c = np.zeros_like(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    end = time.perf_counter()
    return c, end - start

def main():
    print("🔄 Soma de Vetores Serial")
    print("=" * 30)
    
    # Teste com diferentes tamanhos
    sizes = [100_000, 1_000_000, 5_000_000]
    
    for N in sizes:
        print(f"\nTamanho: {N:,} elementos")
        
        # Criar vetores
        a = np.arange(N, dtype=np.float64)
        b = np.arange(N, dtype=np.float64)
        
        # Soma serial
        c, time_taken = vector_sum_serial(a, b)
        
        print(f"Tempo serial: {time_taken:.4f}s")
        print(f"Resultado[0:5]: {c[:5]}")
        print(f"Resultado[-5:]: {c[-5:]}")

if __name__ == "__main__":
    main()