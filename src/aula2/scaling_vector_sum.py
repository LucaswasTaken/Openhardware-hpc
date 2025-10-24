#!/usr/bin/env python3
"""
Exemplo 9: Análise de escalabilidade de operações vetoriais
Demonstra strong e weak scaling para compreender limites de paralelização.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

def vector_operation_worker(args):
    """Worker para operações vetoriais complexas"""
    start_idx, end_idx, data = args
    # Simular operação complexa: múltiplas operações matemáticas
    chunk = data[start_idx:end_idx]
    result = np.sum(chunk**2) + np.sum(np.sin(chunk)) + np.sum(np.cos(chunk))
    return result

def scaling_study_strong(vector_size, max_processes):
    """
    Strong scaling: tamanho fixo, aumentar processos
    """
    print(f"\n📊 Strong Scaling - Tamanho fixo: {vector_size:,} elementos")
    print("-" * 60)
    
    # Gerar dados
    np.random.seed(42)
    data = np.random.randn(vector_size)
    
    results = []
    
    for n_proc in range(1, max_processes + 1):
        # Dividir trabalho
        chunk_size = vector_size // n_proc
        
        start_time = time.perf_counter()
        
        if n_proc == 1:
            # Serial
            total = vector_operation_worker((0, vector_size, data))
        else:
            # Paralelo
            with ProcessPoolExecutor(max_workers=n_proc) as executor:
                futures = []
                for i in range(n_proc):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < n_proc - 1 else vector_size
                    future = executor.submit(vector_operation_worker, (start_idx, end_idx, data))
                    futures.append(future)
                
                total = sum(future.result() for future in as_completed(futures))
        
        exec_time = time.perf_counter() - start_time
        
        if n_proc == 1:
            serial_time = exec_time
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = serial_time / exec_time
            efficiency = speedup / n_proc
        
        results.append({
            'processes': n_proc,
            'time': exec_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"  {n_proc:2d} processos: {exec_time:.4f}s, speedup: {speedup:.2f}x, eficiência: {efficiency:.2f}")
    
    return results

def scaling_study_weak(base_size_per_process, max_processes):
    """
    Weak scaling: trabalho por processo fixo, aumentar processos
    """
    print(f"\n📈 Weak Scaling - {base_size_per_process:,} elementos por processo")
    print("-" * 60)
    
    results = []
    serial_time = None
    
    for n_proc in range(1, max_processes + 1):
        vector_size = base_size_per_process * n_proc
        
        # Gerar dados
        np.random.seed(42)
        data = np.random.randn(vector_size)
        
        start_time = time.perf_counter()
        
        if n_proc == 1:
            # Serial
            total = vector_operation_worker((0, vector_size, data))
            serial_time = time.perf_counter() - start_time
            speedup = 1.0
            efficiency = 1.0
        else:
            # Paralelo
            chunk_size = vector_size // n_proc
            
            with ProcessPoolExecutor(max_workers=n_proc) as executor:
                futures = []
                for i in range(n_proc):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < n_proc - 1 else vector_size
                    future = executor.submit(vector_operation_worker, (start_idx, end_idx, data))
                    futures.append(future)
                
                total = sum(future.result() for future in as_completed(futures))
            
            exec_time = time.perf_counter() - start_time
            # Para weak scaling, eficiência ideal é tempo constante
            efficiency = serial_time / exec_time if exec_time > 0 else 0
            speedup = efficiency * n_proc
        
        if n_proc > 1:
            exec_time = time.perf_counter() - start_time
        else:
            exec_time = serial_time
        
        results.append({
            'processes': n_proc,
            'problem_size': vector_size,
            'time': exec_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"  {n_proc:2d} processos ({vector_size:7,} elementos): {exec_time:.4f}s, eficiência: {efficiency:.2f}")
    
    return results

def plot_scaling_results(strong_results, weak_results):
    """Criar gráficos de escalabilidade"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Strong scaling - Speedup
    processes = [r['processes'] for r in strong_results]
    speedups = [r['speedup'] for r in strong_results]
    ideal_speedup = processes
    
    ax1.plot(processes, speedups, 'bo-', label='Speedup Real', linewidth=2, markersize=8)
    ax1.plot(processes, ideal_speedup, 'r--', label='Speedup Ideal', linewidth=2)
    ax1.set_xlabel('Número de Processos')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Strong Scaling - Speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(processes)
    
    # Strong scaling - Eficiência
    efficiencies = [r['efficiency'] for r in strong_results]
    
    ax2.plot(processes, efficiencies, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Número de Processos')
    ax2.set_ylabel('Eficiência')
    ax2.set_title('Strong Scaling - Eficiência')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(processes)
    ax2.set_ylim(0, 1.1)
    
    # Weak scaling - Tempo
    weak_processes = [r['processes'] for r in weak_results]
    weak_times = [r['time'] for r in weak_results]
    
    ax3.plot(weak_processes, weak_times, 'mo-', linewidth=2, markersize=8)
    ax3.axhline(y=weak_times[0], color='r', linestyle='--', alpha=0.7, label='Tempo Ideal')
    ax3.set_xlabel('Número de Processos')
    ax3.set_ylabel('Tempo de Execução (s)')
    ax3.set_title('Weak Scaling - Tempo')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(weak_processes)
    
    # Weak scaling - Eficiência
    weak_efficiencies = [r['efficiency'] for r in weak_results]
    
    ax4.plot(weak_processes, weak_efficiencies, 'co-', linewidth=2, markersize=8)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Número de Processos')
    ax4.set_ylabel('Eficiência')
    ax4.set_title('Weak Scaling - Eficiência')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(weak_processes)
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🔬 Análise de Escalabilidade de Operações Vetoriais")
    print("=" * 60)
    
    max_proc = min(mp.cpu_count(), 8)  # Limitar para análise
    print(f"Sistema: {mp.cpu_count()} núcleos, testando até {max_proc} processos")
    
    # Strong scaling
    strong_results = scaling_study_strong(1_000_000, max_proc)
    
    # Weak scaling
    weak_results = scaling_study_weak(250_000, max_proc)
    
    # Análise dos resultados
    print("\n📋 Resumo da Análise:")
    print(f"• Strong scaling máximo: {max(r['speedup'] for r in strong_results):.2f}x")
    print(f"• Eficiência final (strong): {strong_results[-1]['efficiency']:.2f}")
    print(f"• Eficiência final (weak): {weak_results[-1]['efficiency']:.2f}")
    
    # Detectar overhead
    overhead_start = None
    for i, result in enumerate(strong_results):
        if result['efficiency'] < 0.8:  # Eficiência abaixo de 80%
            overhead_start = result['processes']
            break
    
    if overhead_start:
        print(f"• Overhead significativo a partir de {overhead_start} processos")
    else:
        print("• Escalabilidade boa em todos os níveis testados")
    
    print("\n💡 Interpretação:")
    print("• Strong scaling: performance com problema fixo")
    print("• Weak scaling: performance com trabalho/processo fixo")
    print("• Eficiência ideal = 1.0 (100%)")
    print("• Overhead reduz eficiência para muitos processos")
    
    print("\n📊 Gerando gráficos...")
    try:
        plot_scaling_results(strong_results, weak_results)
        print("✓ Gráficos salvos como 'scaling_analysis.png'")
    except Exception as e:
        print(f"⚠️  Erro ao gerar gráficos: {e}")
    
    print("\n🎯 Aplicações em Engenharia:")
    print("• Dimensionamento de clusters computacionais")
    print("• Escolha do número ideal de processos")
    print("• Identificação de gargalos de paralelização")
    print("• Previsão de performance em sistemas maiores")

if __name__ == "__main__":
    main()