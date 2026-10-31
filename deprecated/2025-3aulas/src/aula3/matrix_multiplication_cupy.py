#!/usr/bin/env python3
"""
Exemplo 11: Multiplicação de matrizes massivas com CuPy
Demonstra onde GPUs realmente brilham: operações matriciais grandes.
"""

import time
import numpy as np

# Verificar disponibilidade do CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"✅ CuPy {cp.__version__} disponível")
    print(f"   GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️  CuPy não disponível - instale com: pip install cupy-cuda11x")

def matrix_multiply_benchmark(sizes):
    """Benchmark multiplicação de matrizes CPU vs GPU"""
    if not CUPY_AVAILABLE:
        print("CuPy não disponível para benchmark")
        return
    
    print("\n🧮 Benchmark: Multiplicação de Matrizes CPU vs GPU")
    print("=" * 70)
    print("Tamanho      CPU (s)    GPU (s)   Transfer (s)   CPU GFLOPS   GPU GFLOPS   Speedup")
    print("-" * 80)
    
    for size in sizes:
        print(f"\nProcessando matrizes {size}x{size}...")
        
        # Criar matrizes aleatórias
        np.random.seed(42)
        A_cpu = np.random.randn(size, size).astype(np.float32)
        B_cpu = np.random.randn(size, size).astype(np.float32)
        
        # NumPy (CPU) com BLAS otimizado
        start = time.perf_counter()
        C_cpu = np.dot(A_cpu, B_cpu)
        time_cpu = time.perf_counter() - start
        
        # Transferir para GPU
        start = time.perf_counter()
        A_gpu = cp.asarray(A_cpu)
        B_gpu = cp.asarray(B_cpu)
        time_transfer_to = time.perf_counter() - start
        
        # CuPy (GPU)
        start = time.perf_counter()
        C_gpu = cp.dot(A_gpu, B_gpu)
        cp.cuda.Stream.null.synchronize()
        time_gpu = time.perf_counter() - start
        
        # Transferir resultado de volta
        start = time.perf_counter()
        C_gpu_cpu = cp.asnumpy(C_gpu)
        time_transfer_back = time.perf_counter() - start
        
        # Verificar precisão
        max_error = np.max(np.abs(C_cpu - C_gpu_cpu))
        are_close = np.allclose(C_cpu, C_gpu_cpu, rtol=1e-4)
        
        # Calcular métricas
        flops = 2 * size**3  # Multiplicação de matrizes: 2*n³ operações
        gflops_cpu = flops / (time_cpu * 1e9)
        gflops_gpu = flops / (time_gpu * 1e9)
        speedup_compute = time_cpu / time_gpu
        total_transfer = time_transfer_to + time_transfer_back
        
        status = "✓" if are_close else "✗"
        print(f"{size:>6}x{size:<6} {time_cpu:>7.3f}   {time_gpu:>6.3f}   {total_transfer:>9.3f}   {gflops_cpu:>9.1f}   {gflops_gpu:>9.1f}   {speedup_compute:>6.1f}x {status}")
        
        if not are_close:
            print(f"           ⚠️  Erro máximo: {max_error:.2e}")
        
        # Limpeza de memória GPU
        del A_gpu, B_gpu, C_gpu
        cp.get_default_memory_pool().free_all_blocks()

def memory_bandwidth_test():
    """Testa largura de banda de memória"""
    if not CUPY_AVAILABLE:
        return
    
    print("\n📊 Teste de Largura de Banda de Memória")
    print("=" * 45)
    
    sizes_mb = [10, 100, 1000]  # Tamanhos em MB
    
    for size_mb in sizes_mb:
        n_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes por float32
        
        # Criar dados
        data_cpu = np.random.randn(n_elements).astype(np.float32)
        
        # Host → Device
        start = time.perf_counter()
        data_gpu = cp.asarray(data_cpu)
        time_h2d = time.perf_counter() - start
        bandwidth_h2d = size_mb / time_h2d / 1024  # GB/s
        
        # Device → Host
        start = time.perf_counter()
        data_back = cp.asnumpy(data_gpu)
        time_d2h = time.perf_counter() - start
        bandwidth_d2h = size_mb / time_d2h / 1024  # GB/s
        
        # Operação na GPU (memória interna)
        start = time.perf_counter()
        result_gpu = cp.sum(data_gpu**2)
        cp.cuda.Stream.null.synchronize()
        time_gpu_compute = time.perf_counter() - start
        bandwidth_internal = (size_mb * 2) / time_gpu_compute / 1024  # Leitura + escrita
        
        print(f"{size_mb:>4} MB: H→D {bandwidth_h2d:>6.1f} GB/s, D→H {bandwidth_d2h:>6.1f} GB/s, GPU {bandwidth_internal:>6.1f} GB/s")
        
        # Limpeza
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()

def demonstrate_advanced_operations():
    """Demonstra operações avançadas com CuPy"""
    if not CUPY_AVAILABLE:
        return
    
    print("\n🔬 Operações Avançadas com CuPy")
    print("=" * 40)
    
    size = 2048
    
    # Criar matriz complexa
    print("Criando matriz complexa...")
    A = cp.random.randn(size, size).astype(cp.float32)
    
    # Decomposição SVD
    print("Executando SVD...")
    start = time.perf_counter()
    U, s, Vt = cp.linalg.svd(A)
    cp.cuda.Stream.null.synchronize()
    time_svd = time.perf_counter() - start
    print(f"  SVD {size}x{size}: {time_svd:.3f}s")
    
    # Autovalores
    print("Calculando autovalores...")
    start = time.perf_counter()
    eigenvals = cp.linalg.eigvals(A @ A.T)  # Matriz simétrica
    cp.cuda.Stream.null.synchronize()
    time_eig = time.perf_counter() - start
    print(f"  Autovalores: {time_eig:.3f}s")
    
    # FFT
    print("Transformada de Fourier...")
    start = time.perf_counter()
    fft_result = cp.fft.fft2(A)
    cp.cuda.Stream.null.synchronize()
    time_fft = time.perf_counter() - start
    print(f"  FFT 2D: {time_fft:.3f}s")
    
    # Estatísticas
    print(f"\nEstatísticas da matriz:")
    print(f"  Posto: {cp.linalg.matrix_rank(A)}")
    print(f"  Norma Frobenius: {cp.linalg.norm(A, 'fro'):.3f}")
    print(f"  Número de condição: {cp.linalg.cond(A):.3e}")
    
    # Limpeza
    del A, U, s, Vt, eigenvals, fft_result
    cp.get_default_memory_pool().free_all_blocks()

def main():
    if not CUPY_AVAILABLE:
        print("Este exemplo requer CuPy. Instale com:")
        print("  pip install cupy-cuda11x  # Para CUDA 11.x")
        print("  pip install cupy-cuda12x  # Para CUDA 12.x")
        return
    
    print("🚀 Multiplicação de Matrizes Massivas com CuPy")
    print("=" * 55)
    
    # Tamanhos para benchmark
    sizes = [1024, 2048, 4096]
    
    # Executar benchmarks
    matrix_multiply_benchmark(sizes)
    
    # Teste de largura de banda
    memory_bandwidth_test()
    
    # Operações avançadas
    demonstrate_advanced_operations()
    
    print("\n💡 Observações sobre Performance GPU:")
    print("• GPUs brilham em operações matriciais grandes")
    print("• Speedup aumenta com tamanho do problema")
    print("• Largura de banda de memória é crucial")
    print("• Transferências CPU↔GPU são gargalo para dados pequenos")
    
    print("\n📈 Quando usar GPU para matrizes:")
    print("• Matrizes ≥ 1000x1000 elementos")
    print("• Múltiplas operações na mesma sessão")
    print("• Dados já residem na GPU")
    print("• Operações custosas (SVD, eigenvalues)")
    
    print("\n🎯 Aplicações em Engenharia:")
    print("• Análise modal de estruturas grandes")
    print("• Solução de sistemas lineares massivos")
    print("• Decomposições matriciais para FEM")
    print("• Processamento de sinais multicanal")

if __name__ == "__main__":
    main()