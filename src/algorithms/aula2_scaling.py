"""
Aula 2 Scaling Analysis - Advanced Parallelism
Wrapper para análise de escalabilidade usando scripts externos
"""

import subprocess
import sys
import os
import json
import tempfile
import matplotlib.pyplot as plt
import numpy as np


def run_scaling_analysis_external(vector_size=1000000, base_size=250000, process_counts=[1, 2, 4], output_file=None):
    """
    Executa análise de escalabilidade via script externo
    
    Args:
        vector_size: Tamanho do vetor para strong scaling
        base_size: Tamanho base por processo para weak scaling
        process_counts: Lista de números de processos para testar
    
    Returns:
        dict: Resultados da análise de escalabilidade
    """
    print("🔬 Análise de Escalabilidade Real (Script Externo)")
    print("=" * 50)
    
    # Path to multiprocessing script
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aula2_multiprocessing')
    scaling_script = os.path.join(script_dir, 'scaling_analysis_mp.py')
    
    if not os.path.exists(scaling_script):
        print(f"❌ Script não encontrado em: {scaling_script}")
        return None
    
    # Criar arquivo temporário para resultados
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Preparar comando
        process_list = ','.join(map(str, process_counts))
        cmd = [
            sys.executable, scaling_script,
            '--vector-size', str(vector_size),
            '--base-size', str(base_size),
            '--processes', process_list,
            '--output', temp_filename
        ]
        
        # Executar script externo
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd=script_dir, timeout=300)  # 5min timeout
        
        if result.returncode == 0:
            print("   ⚡ Usando multiprocessing REAL (fora do Jupyter)")
            print(result.stdout)
            
            # Carregar resultados
            try:
                with open(temp_filename, 'r') as f:
                    scaling_results = json.load(f)
                return scaling_results
            except (FileNotFoundError, json.JSONDecodeError):
                print("⚠️ Erro ao carregar resultados JSON")
                return None
        else:
            print(f"❌ Erro executando script: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏱️ Script demorou muito (>5min)")
        return None
    except Exception as e:
        print(f"❌ Falha na execução: {e}")
        return None
    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(temp_filename)
        except:
            pass


def plot_scaling_results(scaling_results):
    """
    Plotar resultados de escalabilidade
    """
    if not scaling_results:
        print("❌ Sem resultados para plotar")
        return
    
    strong_results = scaling_results.get('strong_scaling', [])
    weak_results = scaling_results.get('weak_scaling', [])
    
    if not strong_results or not weak_results:
        print("❌ Dados incompletos para plotagem")
        return
    
    # Criar figura
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
    plt.show()
    
    return fig


def explain_scaling_concepts():
    """
    Explica conceitos de strong e weak scaling
    """
    print("📖 Conceitos: Strong vs Weak Scaling")
    print("=" * 40)
    
    print("🔹 Strong Scaling:")
    print("  • Problema fixo, aumentar recursos")
    print("  • Speedup = T₁ / Tₚ")
    print("  • Eficiência = Speedup / P")
    print("  • Limitado pela Lei de Amdahl")
    
    print("\n🔹 Weak Scaling:")
    print("  • Trabalho por processo fixo")
    print("  • Problema cresce com recursos")
    print("  • Tempo ideal = constante")
    print("  • Limitado por overhead de comunicação")
    
    print("\n🔹 Fatores que Afetam Escalabilidade:")
    print("  • Overhead de criação de processos")
    print("  • Comunicação entre processos")
    print("  • Sincronização e load balancing")
    print("  • Memória e cache locality")
    
    print("\n🔹 Aplicações em Engenharia:")
    print("  • Simulações de elementos finitos")
    print("  • Análise de grandes datasets")
    print("  • Otimização paramétrica")
    print("  • Processamento de imagens")
    
    return True


def demonstrate_scaling_analysis():
    """
    Demonstra análise de escalabilidade completa
    """
    print("🎯 Demonstração: Análise de Escalabilidade Real")
    print("=" * 50)
    
    # Explicar conceitos
    explain_scaling_concepts()
    
    print("\n" + "="*50)
    
    # Executar análise
    import multiprocessing as mp
    max_proc = min(mp.cpu_count(), 8)
    process_counts = [1, 2, 4, max_proc] if max_proc > 4 else [1, 2, 4]
    
    scaling_results = run_scaling_analysis_external(
        vector_size=500000,     # Tamanho otimizado para speedup real
        base_size=125000,       # Base otimizada  
        process_counts=[1, 2, 4]  # Processos testados
    )
    
    if scaling_results:
        print("\n📊 Plotando resultados...")
        plot_scaling_results(scaling_results)
        
        print("\n✅ Análise concluída com sucesso!")
        
        # Análise dos resultados
        strong_results = scaling_results['strong_scaling']
        weak_results = scaling_results['weak_scaling']
        
        best_strong_speedup = max(r['speedup'] for r in strong_results[1:])
        best_strong_efficiency = max(r['efficiency'] for r in strong_results[1:])
        
        print(f"\n📈 Melhores Resultados:")
        print(f"  Strong Scaling: {best_strong_speedup:.2f}x speedup")
        print(f"  Melhor Eficiência: {best_strong_efficiency:.2f}")
        
        return scaling_results
    else:
        print("❌ Falha na análise de escalabilidade")
        return None