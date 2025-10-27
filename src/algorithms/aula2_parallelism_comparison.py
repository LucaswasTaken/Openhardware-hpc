"""
Aula 2 - Parallelism Methods Comparison Module
Demonstra comparação entre Threading, Multiprocessing e Joblib
Funciona fora do ambiente Jupyter para evitar BrokenProcessPool
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from joblib import Parallel, delayed
import multiprocessing as mp

def io_bound_task_fast(task_id, duration=0.1):
    """Simula tarefa I/O-bound - versão rápida"""
    time.sleep(duration)
    return f"Task {task_id} completed"

def cpu_bound_task_fast(task_id, n=80_000):
    """Tarefa CPU-bound - versão reduzida"""
    total = 0.0
    for i in range(n):
        total += (i * i) ** 0.5 + np.sin(i * 0.01)
    return total

def cpu_intensive_task(task_id, n=1_000_000):
    """Tarefa CPU muito intensiva - onde multiprocessing ganha"""
    import math
    total = 0.0
    
    # Algoritmo mais pesado: simulação de séries matemáticas
    for i in range(n):
        # Operações matemáticas custosas
        x = float(i)
        total += math.sqrt(x * x + 1.0)
        total += math.sin(x * 0.001) * math.cos(x * 0.001)
        total += math.exp(-x / n) if x / n < 10 else 0.0
        
        # Cálculo de fatorial módulo para números pequenos
        if i % 10000 == 0 and i > 0:
            factorial_sum = 1.0
            for j in range(1, min(15, i % 20 + 5)):
                factorial_sum *= j
            total += math.log(factorial_sum) if factorial_sum > 0 else 0.0
            
        # Série harmônica parcial para aumentar computação
        if i % 5000 == 0 and i > 0:
            harmonic = sum(1.0/k for k in range(1, min(100, i % 50 + 10)))
            total += harmonic
    
    return total

def benchmark_serial(task_func, n_tasks, **kwargs):
    """Execução serial (baseline)"""
    start = time.perf_counter()
    results = [task_func(i, **kwargs) for i in range(n_tasks)]
    return time.perf_counter() - start, results

def benchmark_threading(task_func, n_tasks, n_threads=4, **kwargs):
    """Execução com ThreadPoolExecutor"""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(task_func, i, **kwargs) for i in range(n_tasks)]
        results = [f.result() for f in futures]
    return time.perf_counter() - start, results

def benchmark_multiprocessing(task_func, n_tasks, n_processes=4, **kwargs):
    """Execução com ProcessPoolExecutor - REAL multiprocessing"""
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(task_func, i, **kwargs) for i in range(n_tasks)]
        results = [f.result() for f in futures]
    return time.perf_counter() - start, results

def benchmark_joblib(task_func, n_tasks, n_jobs=4, **kwargs):
    """Execução com Joblib"""
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(task_func)(i, **kwargs) for i in range(n_tasks)
    )
    return time.perf_counter() - start, results

def run_parallelism_comparison():
    """
    Executa comparação completa entre métodos de paralelização
    Retorna resultados para análise
    """
    print("🏁 Comparação Real: Threading vs Multiprocessing vs Joblib")
    print("=" * 65)
    
    # Configurações otimizadas
    n_tasks_io = 12
    io_duration = 0.1
    n_tasks_cpu = 6  # Reduzido para compensar maior intensidade
    cpu_intensity = 1_000_000  # Muito aumentado para garantir vitória do multiprocessing
    
    results = {}
    
    # =============================================================================
    # EXPERIMENTO 1: Tarefas I/O-bound
    # =============================================================================
    
    print("\n🌐 EXPERIMENTO 1: Tarefas I/O-bound")
    print("="*50)
    print(f"📊 Executando {n_tasks_io} tarefas I/O (sleep {io_duration}s cada)...")
    
    # Serial
    print("  🔄 Executando serial...")
    time_serial_io, _ = benchmark_serial(io_bound_task_fast, n_tasks_io, duration=io_duration)
    
    # Threading
    print("  🧵 Executando threading...")
    time_threading_io, _ = benchmark_threading(io_bound_task_fast, n_tasks_io, duration=io_duration)
    
    # Joblib
    print("  ⚡ Executando joblib...")
    time_joblib_io, _ = benchmark_joblib(io_bound_task_fast, n_tasks_io, duration=io_duration)
    
    # Speedups
    speedup_threading_io = time_serial_io / time_threading_io
    speedup_joblib_io = time_serial_io / time_joblib_io
    
    print(f"\n📈 Resultados I/O-bound:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔄 Serial:        {time_serial_io:.3f}s")
    print(f"🧵 Threading:     {time_threading_io:.3f}s → {speedup_threading_io:.1f}x speedup")
    print(f"⚡ Joblib:        {time_joblib_io:.3f}s → {speedup_joblib_io:.1f}x speedup")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    results['io'] = {
        'serial': time_serial_io,
        'threading': time_threading_io,
        'joblib': time_joblib_io,
        'speedup_threading': speedup_threading_io,
        'speedup_joblib': speedup_joblib_io
    }
    
    # =============================================================================
    # EXPERIMENTO 2: Tarefas CPU-bound (Intensivas)
    # =============================================================================
    
    print("\n💻 EXPERIMENTO 2: Tarefas CPU-bound INTENSIVAS")
    print("="*50)
    print(f"📊 Executando {n_tasks_cpu} tarefas CPU MUITO intensivas...")
    print("   • Algoritmo: séries matemáticas, exponenciais, logaritmos, harmônicas")
    print("   • 1M+ operações por tarefa")
    print("   • Objetivo: demonstrar onde multiprocessing VENCE threading")
    
    # Serial
    print("  🔄 Executando serial...")
    time_serial_cpu, _ = benchmark_serial(cpu_intensive_task, n_tasks_cpu, n=cpu_intensity)
    
    # Threading (mostra limitação do GIL)
    print("  🧵 Executando threading...")
    time_threading_cpu, _ = benchmark_threading(cpu_intensive_task, n_tasks_cpu, n=cpu_intensity)
    
    # Multiprocessing (REAL - fora do Jupyter!)
    print("  🔀 Executando REAL multiprocessing...")
    time_multiproc_cpu, _ = benchmark_multiprocessing(cpu_intensive_task, n_tasks_cpu, n=cpu_intensity)
    
    # Joblib
    print("  ⚡ Executando joblib...")
    time_joblib_cpu, _ = benchmark_joblib(cpu_intensive_task, n_tasks_cpu, n=cpu_intensity)
    
    # Speedups
    speedup_threading_cpu = time_serial_cpu / time_threading_cpu
    speedup_multiproc_cpu = time_serial_cpu / time_multiproc_cpu
    speedup_joblib_cpu = time_serial_cpu / time_joblib_cpu
    
    print(f"\n📈 Resultados CPU-bound:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔄 Serial:          {time_serial_cpu:.3f}s")
    print(f"🧵 Threading:       {time_threading_cpu:.3f}s → {speedup_threading_cpu:.2f}x speedup")
    print(f"🔀 Multiprocessing: {time_multiproc_cpu:.3f}s → {speedup_multiproc_cpu:.1f}x speedup")
    print(f"⚡ Joblib:          {time_joblib_cpu:.3f}s → {speedup_joblib_cpu:.1f}x speedup")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    results['cpu'] = {
        'serial': time_serial_cpu,
        'threading': time_threading_cpu,
        'multiprocessing': time_multiproc_cpu,
        'joblib': time_joblib_cpu,
        'speedup_threading': speedup_threading_cpu,
        'speedup_multiprocessing': speedup_multiproc_cpu,
        'speedup_joblib': speedup_joblib_cpu
    }
    
    # =============================================================================
    # ANÁLISE EDUCATIVA
    # =============================================================================
    
    print("\n🎯 ANÁLISE EDUCATIVA: Resultados REAIS")
    print("="*60)
    
    print("\n🧵 THREADING:")
    print("✅ EXCELENTE para I/O-bound tasks")
    print(f"📊 I/O Speedup: {speedup_threading_io:.1f}x (supera latência de rede/disco)")
    print(f"📊 CPU Speedup: {speedup_threading_cpu:.2f}x (limitado pelo GIL)")
    print("💡 Use para: APIs, downloads, operações de arquivo")
    
    print("\n🔀 MULTIPROCESSING (REAL):")
    print("✅ EXCELENTE para CPU-bound tasks")
    print(f"📊 CPU Speedup: {speedup_multiproc_cpu:.1f}x (supera limitação do GIL)")
    if speedup_multiproc_cpu > speedup_threading_cpu:
        print(f"🏆 VENCEU Threading por {speedup_multiproc_cpu/speedup_threading_cpu:.1f}x!")
    print("💡 Use para: cálculos pesados, algoritmos matemáticos")
    
    print("\n⚡ JOBLIB:")
    print("✅ VERSÁTIL e fácil de usar")
    print(f"📊 I/O Speedup: {speedup_joblib_io:.1f}x (backend automático)")
    print(f"📊 CPU Speedup: {speedup_joblib_cpu:.1f}x (escolhe melhor método)")
    print("💡 Use para: data science, machine learning, prototipagem")
    
    print(f"\n🏆 RECOMENDAÇÕES REAIS:")
    print(f"🌐 Para Downloads/APIs → Threading ({speedup_threading_io:.1f}x)")
    if speedup_multiproc_cpu > speedup_threading_cpu:
        print(f"💻 Para Cálculos Pesados → Multiprocessing ({speedup_multiproc_cpu:.1f}x) ✅ VENCE!")
    else:
        print(f"💻 Para Cálculos Pesados → Multiprocessing ({speedup_multiproc_cpu:.1f}x)")
    print(f"📊 Para Data Science → Joblib ({speedup_joblib_cpu:.1f}x)")
    print(f"🎓 Para Iniciantes → Comece com Joblib!")
    
    print("\n✅ Experimento REAL concluído com sucesso!")
    print("🎯 Próximo: GPU computing para speedups de 100-1000x!")
    
    return results

def demonstrate_parallelism_comparison():
    """
    Função para demonstrar conceitos de paralelização
    Chama pela main ou pelo notebook
    """
    print("📚 CONTEXTO: Métodos de Paralelização em Python")
    print("="*60)
    print("• Threading: Ideal para I/O-bound (network, files)")
    print("• Multiprocessing: Ideal para CPU-bound (cálculos)")
    print("• Joblib: Versátil, escolhe automaticamente")
    print("• Windows + Jupyter: ProcessPoolExecutor pode falhar")
    print("• Solução: Executar fora do notebook para resultados reais")
    
    print("\n🔬 EXECUTANDO COMPARAÇÃO REAL...")
    print("="*45)
    
    try:
        results = run_parallelism_comparison()
        return results
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        return None

if __name__ == "__main__":
    # Executar quando chamado diretamente
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparação de Métodos de Paralelização')
    parser.add_argument('--io-tasks', type=int, default=12, help='Número de tarefas I/O')
    parser.add_argument('--cpu-tasks', type=int, default=6, help='Número de tarefas CPU')
    parser.add_argument('--cpu-intensity', type=int, default=80_000, help='Intensidade CPU')
    
    args = parser.parse_args()
    
    print(f"🖥️  Sistema: {mp.cpu_count()} núcleos de CPU")
    print(f"📊 NumPy: {np.__version__}")
    
    results = demonstrate_parallelism_comparison()
    
    if results:
        print(f"\n📋 RESUMO FINAL:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"I/O-bound: Threading {results['io']['speedup_threading']:.1f}x, Joblib {results['io']['speedup_joblib']:.1f}x")
        print(f"CPU-bound: Threading {results['cpu']['speedup_threading']:.2f}x, Multiprocessing {results['cpu']['speedup_multiprocessing']:.1f}x, Joblib {results['cpu']['speedup_joblib']:.1f}x")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")