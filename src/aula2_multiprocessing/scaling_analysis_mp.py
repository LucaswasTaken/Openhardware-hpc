"""
Análise de Escalabilidade Real com Multiprocessing
Estudo de Strong e Weak Scaling para operações vetoriais
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json


def vector_operation_worker(args):
    """Worker para operações vetoriais computacionalmente intensivas"""
    start_idx, end_idx, seed = args
    
    # Gerar dados localmente para evitar transferir arrays grandes
    np.random.seed(42)  # Seed fixo
    data_size = end_idx - start_idx
    data = np.random.randn(data_size)
    
    # Operação MUITO mais intensiva para demonstrar speedup real
    result = 0.0
    
    # Loop intensivo que realmente se beneficia de paralelização
    for iteration in range(5):  # Múltiplas iterações
        # Operações matemáticas intensivas
        result += np.sum(data**3)
        result += np.sum(np.sin(data) * np.cos(data))
        result += np.sum(np.sqrt(np.abs(data) + 1))
        result += np.sum(np.exp(data * 0.01))  # Exponencial (caro)
        
        # Produto escalar entre arrays
        shifted_data = np.roll(data, 1)
        result += np.dot(data, shifted_data)
        
        # Ordenação (operação custosa)
        sorted_data = np.sort(np.abs(data))
        result += np.sum(sorted_data[:100])  # Top 100 elementos
    
    return result


def run_strong_scaling_study(vector_size, process_counts):
    """
    Strong scaling: tamanho fixo, aumentar processos
    """
    print(f"🔬 Strong Scaling - Tamanho fixo: {vector_size:,} elementos")
    print("=" * 60)
    
    results = []
    serial_time = None
    
    for n_proc in process_counts:
        chunk_size = vector_size // n_proc
        
        start_time = time.perf_counter()
        
        if n_proc == 1:
            # Serial
            total = vector_operation_worker((0, vector_size, 42))
            serial_time = time.perf_counter() - start_time
            speedup = 1.0
            efficiency = 1.0
        else:
            # Paralelo
            with ProcessPoolExecutor(max_workers=n_proc) as executor:
                futures = []
                for i in range(n_proc):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < n_proc - 1 else vector_size
                    # Usar seed diferente para cada chunk
                    seed = 42 + i
                    future = executor.submit(vector_operation_worker, (start_idx, end_idx, seed))
                    futures.append(future)
                
                total = sum(future.result() for future in as_completed(futures))
            
            exec_time = time.perf_counter() - start_time
            speedup = serial_time / exec_time
            efficiency = speedup / n_proc
        
        if n_proc == 1:
            exec_time = serial_time
        
        results.append({
            'processes': n_proc,
            'time': exec_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"  {n_proc:2d} processos: {exec_time:.4f}s, speedup: {speedup:.2f}x, eficiência: {efficiency:.2f}")
    
    return results


def run_weak_scaling_study(base_size_per_process, process_counts):
    """
    Weak scaling: trabalho por processo fixo, aumentar processos
    """
    print(f"\n🔬 Weak Scaling - {base_size_per_process:,} elementos por processo")
    print("=" * 60)
    
    results = []
    base_time = None
    
    for n_proc in process_counts:
        vector_size = base_size_per_process * n_proc
        
        start_time = time.perf_counter()
        
        if n_proc == 1:
            # Serial (baseline)
            total = vector_operation_worker((0, vector_size, 42))
            base_time = time.perf_counter() - start_time
            speedup = 1.0
            efficiency = 1.0
        else:
            # Paralelo
            chunk_size = base_size_per_process  # Cada processo trabalha com tamanho fixo
            
            with ProcessPoolExecutor(max_workers=n_proc) as executor:
                futures = []
                for i in range(n_proc):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size
                    seed = 42 + i
                    future = executor.submit(vector_operation_worker, (start_idx, end_idx, seed))
                    futures.append(future)
                
                total = sum(future.result() for future in as_completed(futures))
            
            exec_time = time.perf_counter() - start_time
            # Para weak scaling, o ideal é tempo constante
            speedup = base_time / exec_time
            efficiency = speedup
        
        if n_proc == 1:
            exec_time = base_time
        
        results.append({
            'processes': n_proc,
            'problem_size': vector_size,
            'time': exec_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"  {n_proc:2d} processos ({vector_size:7,} elementos): {exec_time:.4f}s, eficiência: {efficiency:.2f}")
    
    return results


def run_scaling_comparison(vector_size, base_size_per_process, process_counts):
    """Executar ambos os estudos de escalabilidade"""
    
    print(f"🧮 Análise de Escalabilidade Real")
    print(f"Processos testados: {process_counts}")
    print("=" * 60)
    
    # Strong scaling
    strong_results = run_strong_scaling_study(vector_size, process_counts)
    
    # Weak scaling
    weak_results = run_weak_scaling_study(base_size_per_process, process_counts)
    
    # Resumo
    print(f"\n📊 Resumo dos Resultados:")
    print(f"Strong Scaling (problema fixo: {vector_size:,}):")
    for r in strong_results:
        print(f"  {r['processes']} proc: speedup {r['speedup']:.2f}x, eficiência {r['efficiency']:.2f}")
    
    print(f"\nWeak Scaling ({base_size_per_process:,} por processo):")
    for r in weak_results:
        print(f"  {r['processes']} proc: tempo {r['time']:.3f}s, eficiência {r['efficiency']:.2f}")
    
    return {
        'strong_scaling': strong_results,
        'weak_scaling': weak_results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análise de escalabilidade com multiprocessing')
    parser.add_argument('--vector-size', type=int, default=1000000,
                        help='Tamanho do vetor para strong scaling (default: 1000000)')
    parser.add_argument('--base-size', type=int, default=250000,
                        help='Tamanho base por processo para weak scaling (default: 250000)')
    parser.add_argument('--processes', type=str, default='1,2,4', 
                        help='Lista de processos para testar (default: 1,2,4)')
    parser.add_argument('--output', type=str, help='Arquivo JSON para salvar resultados')
    
    args = parser.parse_args()
    
    # Parse process counts
    try:
        process_counts = [int(p.strip()) for p in args.processes.split(',')]
    except ValueError:
        print("❌ Erro: processes deve ser uma lista separada por vírgulas (ex: 1,2,4)")
        exit(1)
    
    # Executar análise
    results = run_scaling_comparison(args.vector_size, args.base_size, process_counts)
    
    # Salvar resultados se especificado
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Resultados salvos em: {args.output}")