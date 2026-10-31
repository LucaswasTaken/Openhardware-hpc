#!/usr/bin/env python3
"""
Exemplo 10: Soma de vetores com CuPy
Demonstra a simplicidade de portar código NumPy para GPU.
"""

import time
import numpy as np

# Verificar disponibilidade do CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"✅ CuPy {cp.__version__} disponível")
    print(f"   GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"   Memória: {cp.cuda.runtime.memGetInfo()[1] // (1024**3)} GB")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️  CuPy não disponível - instale com: pip install cupy-cuda11x")

def benchmark_vector_sum(sizes):
    """Benchmark soma de vetores CPU vs GPU"""
    if not CUPY_AVAILABLE:
        print("CuPy não disponível para benchmark GPU")
        return
    
    print("\n🚀 Benchmark: Soma de Vetores CPU vs GPU")
    print("=" * 55)
    print("Tamanho        NumPy (s)    CuPy (s)    Transfer (s)   Speedup")
    print("-" * 65)
    
    for N in sizes:
        # Criar vetores no CPU
        np.random.seed(42)
        a_cpu = np.random.randn(N).astype(np.float32)
        b_cpu = np.random.randn(N).astype(np.float32)
        
        # NumPy (CPU)
        start = time.perf_counter()
        c_cpu = a_cpu + b_cpu
        time_cpu = time.perf_counter() - start
        
        # Transferir para GPU
        start_transfer = time.perf_counter()
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)
        time_transfer_to = time.perf_counter() - start_transfer
        
        # CuPy (GPU) - computação pura
        start = time.perf_counter()
        c_gpu = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()  # Aguardar conclusão
        time_gpu = time.perf_counter() - start
        
        # Transferir resultado de volta
        start_transfer = time.perf_counter()
        c_gpu_cpu = cp.asnumpy(c_gpu)
        time_transfer_back = time.perf_counter() - start_transfer
        
        # Métricas
        total_transfer = time_transfer_to + time_transfer_back
        speedup_compute = time_cpu / time_gpu if time_gpu > 0 else 0
        
        # Verificar correção
        are_equal = np.allclose(c_cpu, c_gpu_cpu, rtol=1e-5)
        status = "✓" if are_equal else "✗"
        
        print(f"{N:>10,}    {time_cpu:>8.4f}   {time_gpu:>7.4f}    {total_transfer:>9.4f}   {speedup_compute:>6.1f}x {status}")
        
        # Limpeza de memória GPU
        del a_gpu, b_gpu, c_gpu
        cp.get_default_memory_pool().free_all_blocks()

def demonstrate_cupy_api():
    """Demonstra a equivalência da API CuPy vs NumPy"""
    if not CUPY_AVAILABLE:
        print("CuPy não disponível para demonstração")
        return
    
    print("\n📚 Demonstração da API CuPy")
    print("=" * 35)
    
    # Criar dados
    size = 1000
    
    print("NumPy (CPU):")
    print("  a = np.random.randn(1000)")
    print("  b = np.random.randn(1000)")
    print("  c = a + b")
    print("  d = np.sin(c)")
    print("  result = np.sum(d)")
    
    # NumPy
    start = time.perf_counter()
    a_np = np.random.randn(size)
    b_np = np.random.randn(size)
    c_np = a_np + b_np
    d_np = np.sin(c_np)
    result_np = np.sum(d_np)
    time_np = time.perf_counter() - start
    
    print("\nCuPy (GPU) - mudança mínima:")
    print("  a = cp.random.randn(1000)  # np → cp")
    print("  b = cp.random.randn(1000)")
    print("  c = a + b")
    print("  d = cp.sin(c)              # np → cp")
    print("  result = cp.sum(d)         # np → cp")
    
    # CuPy
    start = time.perf_counter()
    a_cp = cp.random.randn(size)
    b_cp = cp.random.randn(size)
    c_cp = a_cp + b_cp
    d_cp = cp.sin(c_cp)
    result_cp = cp.sum(d_cp)
    cp.cuda.Stream.null.synchronize()
    time_cp = time.perf_counter() - start
    
    print(f"\nResultados:")
    print(f"  NumPy:  {result_np:.6f} ({time_np:.4f}s)")
    print(f"  CuPy:   {float(result_cp):.6f} ({time_cp:.4f}s)")
    print(f"  Speedup: {time_np/time_cp:.2f}x")
    
    # Limpeza
    del a_cp, b_cp, c_cp, d_cp, result_cp
    cp.get_default_memory_pool().free_all_blocks()

def memory_management_demo():
    """Demonstra gerenciamento de memória GPU"""
    if not CUPY_AVAILABLE:
        print("CuPy não disponível para demonstração")
        return
    
    print("\n💾 Gerenciamento de Memória GPU")
    print("=" * 40)
    
    # Informações de memória
    mempool = cp.get_default_memory_pool()
    
    def print_memory_info(label):
        free, total = cp.cuda.runtime.memGetInfo()
        used = total - free
        print(f"{label}:")
        print(f"  Total: {total//1024**3:.1f} GB")
        print(f"  Usado: {used//1024**3:.1f} GB")
        print(f"  Livre: {free//1024**3:.1f} GB")
        print(f"  Pool:  {mempool.used_bytes()//1024**3:.1f} GB")
    
    print_memory_info("Estado inicial")
    
    # Alocar arrays grandes
    print("\nAlocando arrays grandes...")
    big_arrays = []
    for i in range(3):
        arr = cp.random.randn(10_000_000).astype(cp.float32)  # ~40MB cada
        big_arrays.append(arr)
    
    print_memory_info("Após alocação")
    
    # Liberar explicitamente
    print("\nLiberando memória...")
    del big_arrays
    mempool.free_all_blocks()
    
    print_memory_info("Após liberação")
    
    print("\n💡 Dicas de gerenciamento:")
    print("• Use cp.get_default_memory_pool().free_all_blocks()")
    print("• del variáveis grandes quando não precisar mais")
    print("• Monitore uso com cp.cuda.runtime.memGetInfo()")
    print("• Considere usar context managers para limpeza automática")

def main():
    if not CUPY_AVAILABLE:
        print("Este exemplo requer CuPy. Instale com:")
        print("  pip install cupy-cuda11x  # Para CUDA 11.x")
        print("  pip install cupy-cuda12x  # Para CUDA 12.x")
        return
    
    print("⚡ Soma de Vetores com CuPy")
    print("=" * 35)
    
    # Tamanhos para benchmark
    sizes = [1_000_000, 10_000_000, 100_000_000]
    
    # Executar benchmarks
    benchmark_vector_sum(sizes)
    
    # Demonstrar API
    demonstrate_cupy_api()
    
    # Gerenciamento de memória
    memory_management_demo()
    
    print("\n🎯 Pontos-chave sobre CuPy:")
    print("• API quase idêntica ao NumPy (np → cp)")
    print("• Transferência CPU↔GPU pode ser custosa")
    print("• Melhor performance para operações grandes")
    print("• Gerenciamento de memória GPU é importante")
    print("• Ideal para algoritmos já vetorizados")
    
    print("\n🚀 Aplicações em Engenharia:")
    print("• Processamento de sinais sísmicos")
    print("• Análise de imagens de inspeção")
    print("• Operações matriciais em FEM")
    print("• Simulações Monte Carlo")

if __name__ == "__main__":
    main()