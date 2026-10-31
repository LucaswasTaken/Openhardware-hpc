#!/usr/bin/env python3
"""
Exemplo 8: Multiplicação de matrizes com Numba
Demonstra o poder da compilação JIT e paralelismo automático.
"""

import time
import numpy as np

# Verificar disponibilidade do Numba
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
    print(f"✅ Numba {numba.__version__} disponível")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba não disponível - instale com: pip install numba")

if NUMBA_AVAILABLE:
    def matrix_mult_python(A, B):
        """Multiplicação de matrizes em Python puro (lento)"""
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape
        
        C = np.zeros((rows_A, cols_B))
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    @jit(nopython=True)
    def matrix_mult_numba_serial(A, B):
        """Multiplicação de matrizes com Numba (serial)"""
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape
        
        C = np.zeros((rows_A, cols_B))
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    @jit(nopython=True, parallel=True)
    def matrix_mult_numba_parallel(A, B):
        """Multiplicação de matrizes com Numba paralelo"""
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape
        
        C = np.zeros((rows_A, cols_B))
        for i in prange(rows_A):  # prange = parallel range
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i, j] += A[i, k] * B[k, j]
        return C

def matrix_multiply_numpy(A, B):
    """Multiplicação usando NumPy (baseline)"""
    start = time.perf_counter()
    C = np.dot(A, B)
    end = time.perf_counter()
    return C, end - start

def main():
    if not NUMBA_AVAILABLE:
        print("Este exemplo requer Numba. Instale com: pip install numba")
        return
    
    print("🚀 Multiplicação de Matrizes com Numba")
    print("=" * 45)
    
    # Testar com diferentes tamanhos
    sizes = [256, 512, 1024]
    
    print("Tamanho    NumPy (s)    Python (s)   Numba S (s)   Numba P (s)   Speedup")
    print("-" * 75)
    
    for size in sizes:
        print(f"\nTestando matrizes {size}x{size}...")
        
        # Criar matrizes aleatórias
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float64)
        B = np.random.randn(size, size).astype(np.float64)
        
        # NumPy (baseline)
        C_numpy, time_numpy = matrix_multiply_numpy(A, B)
        
        # Python puro (apenas para matrizes pequenas)
        if size <= 256:
            start = time.perf_counter()
            C_python = matrix_mult_python(A, B)
            time_python = time.perf_counter() - start
        else:
            time_python = float('inf')  # Muito lento para matrizes grandes
        
        # Numba serial (primeira execução inclui compilação)
        start = time.perf_counter()
        C_numba_serial = matrix_mult_numba_serial(A, B)
        time_numba_serial_first = time.perf_counter() - start
        
        # Numba serial (segunda execução, já compilado)
        start = time.perf_counter()
        C_numba_serial = matrix_mult_numba_serial(A, B)
        time_numba_serial = time.perf_counter() - start
        
        # Numba paralelo (primeira execução)
        start = time.perf_counter()
        C_numba_parallel = matrix_mult_numba_parallel(A, B)
        time_numba_parallel_first = time.perf_counter() - start
        
        # Numba paralelo (segunda execução)
        start = time.perf_counter()
        C_numba_parallel = matrix_mult_numba_parallel(A, B)
        time_numba_parallel = time.perf_counter() - start
        
        # Verificar correção
        numpy_vs_serial = np.allclose(C_numpy, C_numba_serial, rtol=1e-10)
        numpy_vs_parallel = np.allclose(C_numpy, C_numba_parallel, rtol=1e-10)
        
        # Speedups
        speedup_numba_parallel = time_numpy / time_numba_parallel if time_numba_parallel > 0 else 0
        
        # Mostrar resultados
        python_str = f"{time_python:>9.3f}" if time_python != float('inf') else "    >10.0"
        print(f"{size:>6}x{size:<6} {time_numpy:>8.3f}   {python_str}   {time_numba_serial:>9.3f}   {time_numba_parallel:>9.3f}   {speedup_numba_parallel:>6.2f}x")
        
        # Status de verificação
        status_s = "✓" if numpy_vs_serial else "✗"
        status_p = "✓" if numpy_vs_parallel else "✗"
        print(f"           Correção: Serial {status_s}, Paralelo {status_p}")
        
        if size == sizes[0]:  # Mostrar tempos de compilação apenas uma vez
            print(f"           Compilação: Serial {time_numba_serial_first:.3f}s, Paralelo {time_numba_parallel_first:.3f}s")
    
    print("\n💡 Observações sobre Numba:")
    print("• JIT (Just-In-Time) compilation oferece speedups dramáticos")
    print("• Primeira execução inclui tempo de compilação")
    print("• @jit(parallel=True) + prange ativa paralelismo automático")
    print("• Funciona melhor com loops e operações numéricas")
    print("• Speedup varia conforme algoritmo e hardware")
    
    print("\n🎯 Quando usar Numba:")
    print("• Loops intensivos em Python puro")
    print("• Algoritmos não vetorizáveis facilmente")
    print("• Quando NumPy não é suficientemente rápido")
    print("• Algoritmos customizados para problemas específicos")

if __name__ == "__main__":
    main()