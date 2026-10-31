#!/usr/bin/env python3
"""
Exemplo 13: Simulação de difusão de calor 2D
Aplicação real de engenharia: simulação térmica em placas metálicas.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Verificar disponibilidades
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

def heat_equation_cpu(T, alpha, dx, dy, dt, steps):
    """
    Simula difusão de calor 2D no CPU
    Equação: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
    """
    ny, nx = T.shape
    T_new = T.copy()
    
    start = time.perf_counter()
    
    for step in range(steps):
        # Atualizar pontos internos usando diferenças finitas
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T_new[i, j] = T[i, j] + alpha * dt * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
                )
        
        # Trocar arrays
        T, T_new = T_new, T
    
    exec_time = time.perf_counter() - start
    return T, exec_time

if CUPY_AVAILABLE:
    def heat_equation_gpu(T_gpu, alpha, dx, dy, dt, steps):
        """Simula difusão de calor 2D na GPU usando CuPy"""
        ny, nx = T_gpu.shape
        T_new_gpu = cp.copy(T_gpu)
        
        start = time.perf_counter()
        
        for step in range(steps):
            # Atualizar usando slicing vetorizado
            T_new_gpu[1:-1, 1:-1] = T_gpu[1:-1, 1:-1] + alpha * dt * (
                (T_gpu[2:, 1:-1] - 2*T_gpu[1:-1, 1:-1] + T_gpu[:-2, 1:-1]) / dx**2 +
                (T_gpu[1:-1, 2:] - 2*T_gpu[1:-1, 1:-1] + T_gpu[1:-1, :-2]) / dy**2
            )
            
            # Trocar arrays
            T_gpu, T_new_gpu = T_new_gpu, T_gpu
            
        cp.cuda.Stream.null.synchronize()
        exec_time = time.perf_counter() - start
        return T_gpu, exec_time

if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def heat_equation_kernel(T_old, T_new, alpha, dx, dy, dt):
        """Kernel CUDA para um passo da equação de calor"""
        i, j = cuda.grid(2)  # Obter coordenadas 2D
        
        ny, nx = T_old.shape
        
        # Verificar bounds (evitar bordas)
        if 1 <= i < ny-1 and 1 <= j < nx-1:
            T_new[i, j] = T_old[i, j] + alpha * dt * (
                (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / (dx*dx) +
                (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / (dy*dy)
            )
    
    def heat_equation_cuda(T_host, alpha, dx, dy, dt, steps):
        """Simula difusão de calor 2D usando kernels CUDA"""
        ny, nx = T_host.shape
        
        # Alocar na GPU
        T_gpu = cuda.to_device(T_host)
        T_new_gpu = cuda.device_array_like(T_gpu)
        
        # Configurar grid de threads
        threads_per_block = (16, 16)
        blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        start = time.perf_counter()
        
        for step in range(steps):
            heat_equation_kernel[blocks_per_grid, threads_per_block](
                T_gpu, T_new_gpu, alpha, dx, dy, dt
            )
            
            # Trocar buffers
            T_gpu, T_new_gpu = T_new_gpu, T_gpu
        
        cuda.synchronize()
        exec_time = time.perf_counter() - start
        
        # Transferir resultado de volta
        result = T_gpu.copy_to_host()
        return result, exec_time

def create_initial_conditions(nx, ny, scenario="center_heat"):
    """Cria condições iniciais para diferentes cenários"""
    T = np.zeros((ny, nx), dtype=np.float32)
    
    if scenario == "center_heat":
        # Fonte de calor no centro
        center_x, center_y = nx//2, ny//2
        radius = min(nx, ny) // 8
        for i in range(ny):
            for j in range(nx):
                if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                    T[i, j] = 100.0  # 100°C
    
    elif scenario == "corner_heat":
        # Fonte de calor no canto
        T[:ny//4, :nx//4] = 80.0
    
    elif scenario == "line_heat":
        # Linha de calor horizontal
        T[ny//2-2:ny//2+2, nx//4:3*nx//4] = 90.0
    
    # Bordas mantidas a 0°C (condição de Dirichlet)
    T[0, :] = T[-1, :] = 0.0
    T[:, 0] = T[:, -1] = 0.0
    
    return T

def analyze_temperature_evolution(T_initial, T_final, dx, dy):
    """Analisa a evolução da temperatura"""
    ny, nx = T_initial.shape
    
    # Perfil central horizontal
    center_y = ny // 2
    x_coords = np.arange(nx) * dx
    profile_initial = T_initial[center_y, :]
    profile_final = T_final[center_y, :]
    
    # Estatísticas
    max_temp_initial = np.max(T_initial)
    max_temp_final = np.max(T_final)
    total_heat_initial = np.sum(T_initial) * dx * dy
    total_heat_final = np.sum(T_final) * dx * dy
    
    print(f"\n📊 Análise da Evolução Térmica:")
    print(f"  Temperatura máxima inicial: {max_temp_initial:.1f}°C")
    print(f"  Temperatura máxima final:   {max_temp_final:.1f}°C")
    print(f"  Calor total inicial: {total_heat_initial:.3f} J")
    print(f"  Calor total final:   {total_heat_final:.3f} J")
    print(f"  Conservação de energia: {100*total_heat_final/total_heat_initial:.1f}%")
    
    return x_coords, profile_initial, profile_final

def visualize_results(T_initial, T_final, x_coords, profile_initial, profile_final):
    """Visualiza os resultados da simulação"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Estado inicial
    im1 = axes[0, 0].imshow(T_initial, cmap='hot', interpolation='bilinear')
    axes[0, 0].set_title('Estado Inicial (t = 0)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0], label='Temperatura (°C)')
    
    # Estado final
    im2 = axes[0, 1].imshow(T_final, cmap='hot', interpolation='bilinear')
    axes[0, 1].set_title('Estado Final')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1], label='Temperatura (°C)')
    
    # Perfis de temperatura
    axes[1, 0].plot(x_coords, profile_initial, 'r-', linewidth=2, label='Inicial')
    axes[1, 0].plot(x_coords, profile_final, 'b-', linewidth=2, label='Final')
    axes[1, 0].set_xlabel('Posição x (m)')
    axes[1, 0].set_ylabel('Temperatura (°C)')
    axes[1, 0].set_title('Perfil de Temperatura (linha central)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Diferença
    diff = T_final - T_initial
    im4 = axes[1, 1].imshow(diff, cmap='RdBu_r', interpolation='bilinear')
    axes[1, 1].set_title('Mudança de Temperatura')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1], label='ΔT (°C)')
    
    plt.tight_layout()
    plt.savefig('heat_diffusion_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🌡️  Simulação de Difusão de Calor 2D")
    print("=" * 45)
    
    # Parâmetros físicos
    Lx, Ly = 1.0, 1.0  # Dimensões da placa (metros)
    nx, ny = 256, 256  # Pontos da malha
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    alpha = 1e-4  # Difusividade térmica (m²/s)
    dt = 0.2 * min(dx, dy)**2 / (4 * alpha)  # Passo de tempo estável
    steps = 300  # Número de passos de tempo
    
    print(f"Parâmetros da simulação:")
    print(f"  Malha: {nx}x{ny} = {nx*ny:,} pontos")
    print(f"  Espaçamento: dx = {dx:.4f}m, dy = {dy:.4f}m")
    print(f"  Passo temporal: dt = {dt:.2e}s")
    print(f"  Passos de tempo: {steps}")
    print(f"  Estabilidade: {dt*alpha/(dx**2):.3f} < 0.25 ✓")
    
    # Condições iniciais
    T_initial = create_initial_conditions(nx, ny, "center_heat")
    print(f"\nCondições: fonte de calor central a 100°C, bordas a 0°C")
    
    # Simular no CPU
    print("\n⏱️  Executando simulação no CPU...")
    T_cpu, time_cpu = heat_equation_cpu(T_initial.copy(), alpha, dx, dy, dt, steps)
    print(f"CPU: {time_cpu:.3f}s")
    
    results = [("CPU", T_cpu, time_cpu)]
    
    # Simular na GPU (CuPy)
    if CUPY_AVAILABLE:
        print("\n⏱️  Executando simulação na GPU (CuPy)...")
        T_gpu_cupy, time_gpu_cupy = heat_equation_gpu(cp.asarray(T_initial), alpha, dx, dy, dt, steps)
        T_gpu_cupy_host = cp.asnumpy(T_gpu_cupy)
        
        speedup_cupy = time_cpu / time_gpu_cupy
        max_diff_cupy = np.max(np.abs(T_cpu - T_gpu_cupy_host))
        
        print(f"GPU (CuPy): {time_gpu_cupy:.3f}s, speedup: {speedup_cupy:.2f}x")
        print(f"Diferença máxima vs CPU: {max_diff_cupy:.2e}°C")
        
        results.append(("GPU (CuPy)", T_gpu_cupy_host, time_gpu_cupy))
    
    # Simular na GPU (Numba CUDA)
    if NUMBA_CUDA_AVAILABLE:
        print("\n⏱️  Executando simulação na GPU (Numba CUDA)...")
        T_gpu_cuda, time_gpu_cuda = heat_equation_cuda(T_initial.copy(), alpha, dx, dy, dt, steps)
        
        speedup_cuda = time_cpu / time_gpu_cuda
        max_diff_cuda = np.max(np.abs(T_cpu - T_gpu_cuda))
        
        print(f"GPU (CUDA): {time_gpu_cuda:.3f}s, speedup: {speedup_cuda:.2f}x")
        print(f"Diferença máxima vs CPU: {max_diff_cuda:.2e}°C")
        
        results.append(("GPU (CUDA)", T_gpu_cuda, time_gpu_cuda))
    
    # Análise dos resultados
    x_coords, profile_initial, profile_final = analyze_temperature_evolution(
        T_initial, T_cpu, dx, dy
    )
    
    # Comparação de performance
    print(f"\n📈 Resumo de Performance:")
    print("Método           Tempo (s)    Speedup")
    print("-" * 35)
    for name, _, exec_time in results:
        speedup = time_cpu / exec_time if exec_time > 0 else 1.0
        print(f"{name:<15} {exec_time:>8.3f}    {speedup:>6.2f}x")
    
    # Visualização
    print(f"\n📊 Gerando visualizações...")
    try:
        visualize_results(T_initial, T_cpu, x_coords, profile_initial, profile_final)
        print("✓ Gráficos salvos como 'heat_diffusion_results.png'")
    except Exception as e:
        print(f"⚠️  Erro na visualização: {e}")
    
    print("\n💡 Aplicações em Engenharia Civil:")
    print("• Análise térmica de estruturas de concreto")
    print("• Simulação de incêndios em edificações")
    print("• Comportamento térmico de pavimentos")
    print("• Isolamento térmico de edifícios")
    print("• Pontes térmicas em estruturas")
    print("• Cura térmica do concreto")
    
    print("\n🎯 Extensões possíveis:")
    print("• Materiais com propriedades variáveis")
    print("• Condições de contorno complexas")
    print("• Acoplamento termo-mecânico")
    print("• Geometrias 3D e malhas não-uniformes")

if __name__ == "__main__":
    main()