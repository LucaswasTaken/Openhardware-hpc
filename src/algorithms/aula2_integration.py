"""
Aula 2 Algorithms - Advanced Parallelism
Contém algoritmos que demonstram paralelismo avançado usando scripts externos
"""

import subprocess
import sys
import os
import time
import numpy as np


def run_integration_comparison_external(n_points=50000000, max_processes=None):
    """
    Executa integração numérica paralela via script externo
    
    Args:
        n_points: Número de pontos de integração
        max_processes: Número máximo de processos
    
    Returns:
        dict: Resultados da integração
    """
    print("🧮 Integração Numérica Paralela (Script Externo)")
    print("=" * 50)
    
    # Path to multiprocessing script
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aula2_multiprocessing')
    integration_script = os.path.join(script_dir, 'integration_mp.py')
    
    if os.path.exists(integration_script):
        try:
            # Executar script externo
            cmd = [sys.executable, integration_script, '--points', str(n_points)]
            
            if max_processes:
                # Se max_processes for um número, criar lista
                if isinstance(max_processes, int):
                    process_list = f"1,2,{max_processes}"
                else:
                    process_list = str(max_processes)
                cmd.extend(['--processes', process_list])
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=script_dir, timeout=120)
            
            if result.returncode == 0:
                print(result.stdout)
                return {'success': True, 'output': result.stdout}
            else:
                print(f"❌ Erro: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - reduzindo número de pontos")
            # Retry with fewer points
            return run_integration_comparison_external(n_points//2, max_processes)
        except Exception as e:
            print(f"❌ Falha na execução: {e}")
            return {'success': False, 'error': str(e)}
    else:
        print("❌ Script de integração não encontrado")
        return {'success': False, 'error': 'Script not found'}


def simulate_integration_notebook():
    """
    Simula integração no notebook para demonstração educacional
    Mostra diferença entre threading (limitado) e conceito de multiprocessing
    """
    print("📚 Demonstração Educacional - Limitações do Notebook")
    print("=" * 55)
    
    # Função para integrar
    def function_to_integrate(x):
        return x**2 * np.sin(x) + np.cos(x)
    
    # Integração serial simples
    a, b = 0, 10
    n_points = 100000  # Menor para demonstração
    
    print(f"Integrando f(x) = x²·sin(x) + cos(x) de {a} a {b}")
    print(f"Pontos: {n_points:,} (reduzido para demonstração)")
    
    start_time = time.perf_counter()
    x = np.linspace(a, b, n_points)
    y = function_to_integrate(x)
    h = (b - a) / (n_points - 1)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    exec_time = time.perf_counter() - start_time
    
    print(f"\nSerial (notebook): Integral = {integral:.6f}, Tempo = {exec_time:.4f}s")
    
    print("\n💡 Para multiprocessing real:")
    print("• Use o script externo integration_mp.py")
    print("• ProcessPoolExecutor requer processos separados")
    print("• Jupyter + subprocess = solução que funciona")
    
    return integral, exec_time


def demonstrate_integration_methods():
    """
    Demonstra diferentes métodos de integração e suas limitações
    """
    print("🔄 Comparação: Notebook vs Script Externo")
    print("=" * 45)
    
    # Simulação no notebook
    print("1️⃣ Execução no Notebook:")
    integral_notebook, time_notebook = simulate_integration_notebook()
    
    print(f"\n2️⃣ Execução via Script Externo:")
    # Execução via script externo
    result = run_integration_comparison_external(n_points=50000000, max_processes=4)
    
    print(f"\n📊 Resumo:")
    print(f"• Notebook: Limitado a serial/threading (GIL)")
    print(f"• Script: Multiprocessing real com speedup")
    print(f"• Solução: Híbrida (notebook + subprocess)")
    
    return {
        'notebook': {'integral': integral_notebook, 'time': time_notebook},
        'external': result
    }


def explain_integration_parallelism():
    """
    Explica conceitos de paralelização em integração numérica
    """
    print("📖 Conceitos: Paralelização de Integração Numérica")
    print("=" * 55)
    
    print("🔹 Regra do Trapézio:")
    print("  ∫f(x)dx ≈ h[f(x₀)/2 + f(x₁) + f(x₂) + ... + f(xₙ₋₁) + f(xₙ)/2]")
    
    print("\n🔹 Estratégia de Paralelização:")
    print("  1. Dividir domínio [a,b] em chunks")
    print("  2. Cada processo integra um chunk")
    print("  3. Somar resultados dos chunks")
    
    print("\n🔹 Vantagens:")
    print("  • Embaraçosamente paralelo")
    print("  • Escala bem com número de processos")
    print("  • Útil para funções complexas")
    
    print("\n🔹 Aplicações em Engenharia:")
    print("  • Integração de cargas distribuídas")
    print("  • Cálculo de momentos e centroides")
    print("  • Análise de espectros de resposta")
    print("  • Integração de equações diferenciais")
    
    return True