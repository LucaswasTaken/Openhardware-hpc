#!/usr/bin/env python3
"""
Exemplo 12: Estimativa de π usando Monte Carlo com Numba CUDA
Demonstra programação de kernels CUDA customizados.
"""

import time
import numpy as np

# Verificar disponibilidade do Numba CUDA
try:
    from numba import cuda
    import numba
    NUMBA_CUDA_AVAILABLE = True
    print(f"✅ Numba CUDA {numba.__version__} disponível")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    print("⚠️  Numba CUDA não disponível")

if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def monte_carlo_pi_kernel(rng_states, n_per_thread, results):
        """Kernel para estimativa de π por Monte Carlo"""
        thread_id = cuda.grid(1)
        
        if thread_id >= rng_states.size:
            return
            
        # Cada thread conta pontos dentro do círculo
        count = 0
        for i in range(n_per_thread):
            x = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            y = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            
            if x*x + y*y <= 1.0:
                count += 1
        
        results[thread_id] = count
    
    @cuda.jit
    def vector_add_kernel(a, b, c):
        """Kernel simples para soma de vetores"""
        i = cuda.grid(1)
        
        if i < a.size:
            c[i] = a[i] + b[i]
    
    @cuda.jit
    def complex_math_kernel(input_array, output_array):
        """Kernel com operações matemáticas complexas"""
        i = cuda.grid(1)
        
        if i < input_array.size:
            x = input_array[i]
            # Operações matemáticas variadas
            result = x * x + cuda.libdevice.sinf(x) + cuda.libdevice.cosf(x)
            result += cuda.libdevice.sqrtf(abs(x) + 1.0)
            output_array[i] = result

def monte_carlo_pi_cuda(n_samples, n_threads=256):
    """Estimativa de π usando CUDA"""
    if not NUMBA_CUDA_AVAILABLE:
        return None, 0
    
    n_blocks = (n_samples + n_threads - 1) // n_threads
    samples_per_thread = n_samples // (n_blocks * n_threads)
    
    # Alocar arrays na GPU
    rng_states = cuda.random.create_xoroshiro128p_states(n_blocks * n_threads, seed=42)
    results = cuda.device_array(n_blocks * n_threads, dtype=np.int32)
    
    # Executar kernel
    start = time.perf_counter()
    monte_carlo_pi_kernel[n_blocks, n_threads](rng_states, samples_per_thread, results)
    cuda.synchronize()
    time_gpu = time.perf_counter() - start
    
    # Transferir resultados e somar
    results_host = results.copy_to_host()
    total_inside = np.sum(results_host)
    total_samples = samples_per_thread * n_blocks * n_threads
    
    pi_estimate = 4.0 * total_inside / total_samples
    return pi_estimate, time_gpu

def monte_carlo_pi_cpu(n_samples):
    """Versão CPU para comparação"""
    start = time.perf_counter()
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside = np.sum((x**2 + y**2) <= 1)
    pi_estimate = 4 * inside / n_samples
    time_cpu = time.perf_counter() - start
    return pi_estimate, time_cpu

def demonstrate_kernel_basics():
    """Demonstra conceitos básicos de kernels CUDA"""
    if not NUMBA_CUDA_AVAILABLE:
        print("Numba CUDA não disponível para demonstração")
        return
    
    print("\n🔧 Conceitos Básicos de Kernels CUDA")
    print("=" * 45)
    
    # Exemplo simples: soma de vetores
    size = 1_000_000
    
    # Criar dados no host
    a_host = np.random.randn(size).astype(np.float32)
    b_host = np.random.randn(size).astype(np.float32)
    
    # Alocar na GPU
    a_device = cuda.to_device(a_host)
    b_device = cuda.to_device(b_host)
    c_device = cuda.device_array(size, dtype=np.float32)
    
    # Configurar grid
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    
    print(f"Configuração do grid:")
    print(f"  Threads por bloco: {threads_per_block}")
    print(f"  Blocos no grid: {blocks_per_grid}")
    print(f"  Total de threads: {threads_per_block * blocks_per_grid}")
    
    # Executar kernel
    start = time.perf_counter()
    vector_add_kernel[blocks_per_grid, threads_per_block](a_device, b_device, c_device)
    cuda.synchronize()
    time_kernel = time.perf_counter() - start
    
    # Transferir resultado
    c_host = c_device.copy_to_host()
    
    # Verificar com NumPy
    c_numpy = a_host + b_host
    max_error = np.max(np.abs(c_host - c_numpy))
    
    print(f"\nResultados:")
    print(f"  Tempo kernel: {time_kernel:.4f}s")
    print(f"  Erro máximo: {max_error:.2e}")
    print(f"  Correto: {'✓' if max_error < 1e-6 else '✗'}")

def performance_comparison():
    """Compara performance CPU vs GPU para diferentes operações"""
    if not NUMBA_CUDA_AVAILABLE:
        print("Numba CUDA não disponível para comparação")
        return
    
    print("\n📊 Comparação de Performance CPU vs GPU")
    print("=" * 50)
    
    sizes = [1_000_000, 10_000_000, 50_000_000]
    
    print("Operação: Monte Carlo π")
    print("Amostras      CPU (s)    GPU (s)    Speedup")
    print("-" * 45)
    
    for n_samples in sizes:
        # CPU
        pi_cpu, time_cpu = monte_carlo_pi_cpu(n_samples)
        
        # GPU
        pi_gpu, time_gpu = monte_carlo_pi_cuda(n_samples)
        
        if pi_gpu is not None:
            speedup = time_cpu / time_gpu
            error_cpu = abs(pi_cpu - np.pi)
            error_gpu = abs(pi_gpu - np.pi)
            
            print(f"{n_samples:>9,}   {time_cpu:>7.3f}   {time_gpu:>6.3f}   {speedup:>6.1f}x")
            
            if n_samples == sizes[0]:  # Mostrar precisão apenas uma vez
                print(f"           π_CPU: {pi_cpu:.6f} (erro: {error_cpu:.6f})")
                print(f"           π_GPU: {pi_gpu:.6f} (erro: {error_gpu:.6f})")
        else:
            print(f"{n_samples:>9,}   {time_cpu:>7.3f}      -         -")

def demonstrate_advanced_cuda():
    """Demonstra conceitos avançados de CUDA"""
    if not NUMBA_CUDA_AVAILABLE:
        return
    
    print("\n🚀 Conceitos Avançados de CUDA")
    print("=" * 40)
    
    # Teste com operações matemáticas complexas
    size = 5_000_000
    input_data = np.random.randn(size).astype(np.float32)
    
    # Alocar na GPU
    input_device = cuda.to_device(input_data)
    output_device = cuda.device_array(size, dtype=np.float32)
    
    # Configurar grid
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    
    # Executar kernel
    start = time.perf_counter()
    complex_math_kernel[blocks_per_grid, threads_per_block](input_device, output_device)
    cuda.synchronize()
    time_gpu = time.perf_counter() - start
    
    # Comparar com CPU
    start = time.perf_counter()
    x = input_data
    result_cpu = x*x + np.sin(x) + np.cos(x) + np.sqrt(np.abs(x) + 1.0)
    time_cpu = time.perf_counter() - start
    
    # Verificar resultado
    result_gpu = output_device.copy_to_host()
    max_error = np.max(np.abs(result_gpu - result_cpu))
    speedup = time_cpu / time_gpu
    
    print(f"Operações matemáticas complexas ({size:,} elementos):")
    print(f"  CPU: {time_cpu:.4f}s")
    print(f"  GPU: {time_gpu:.4f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Erro máximo: {max_error:.2e}")
    
    # Informações do dispositivo
    device = cuda.get_current_device()
    print(f"\nInformações do dispositivo:")
    print(f"  Nome: {device.name}")
    print(f"  Compute capability: {device.compute_capability}")
    print(f"  Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    print(f"  Threads por bloco máx: {device.MAX_THREADS_PER_BLOCK}")

def main():
    if not NUMBA_CUDA_AVAILABLE:
        print("Este exemplo requer Numba com suporte CUDA.")
        print("Instale com: pip install numba")
        print("E verifique se CUDA está instalado no sistema.")
        return
    
    print("🎲 Monte Carlo π com Numba CUDA")
    print("=" * 40)
    
    # Demonstrar conceitos básicos
    demonstrate_kernel_basics()
    
    # Comparação de performance
    performance_comparison()
    
    # Conceitos avançados
    demonstrate_advanced_cuda()
    
    print("\n💡 Conceitos-chave CUDA:")
    print("• Grid: coleção de blocos")
    print("• Block: coleção de threads")
    print("• Thread: unidade de execução")
    print("• cuda.grid(1): obtém índice global do thread")
    print("• cuda.synchronize(): aguarda conclusão na GPU")
    
    print("\n🎯 Quando usar Numba CUDA:")
    print("• Algoritmos com paralelismo massivo")
    print("• Operações não disponíveis em CuPy")
    print("• Controle fino sobre execução GPU")
    print("• Algoritmos customizados específicos")
    
    print("\n🚀 Aplicações em Engenharia:")
    print("• Simulações Monte Carlo para confiabilidade")
    print("• Algoritmos de otimização paralelos")
    print("• Processamento de sinais em tempo real")
    print("• Métodos numéricos customizados")

if __name__ == "__main__":
    main()