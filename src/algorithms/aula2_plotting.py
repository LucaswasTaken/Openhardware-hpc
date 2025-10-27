"""
Plotting utilities para Aula 2 - Paralelismo Avançado
Visualizações educativas para speedup, eficiência e overhead analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import multiprocessing as mp

# Configure style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

def plot_speedup_comparison(results_list, title="Comparação de Speedup"):
    """
    Plota comparação de speedup entre diferentes métodos
    
    Args:
        results_list: Lista de dicts com resultados de análises
        title: Título do gráfico
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extrair dados
    methods = [r['analysis_type'] for r in results_list]
    speedups = [r['speedup'] for r in results_list]
    efficiencies = [r['efficiency'] for r in results_list]
    times_serial = [r['time_serial'] for r in results_list]
    times_parallel = [r['time_parallel'] for r in results_list]
    
    colors = ['#ff6b6b' if s < 1.0 else '#4ecdc4' for s in speedups]
    
    # 1. Speedup comparison
    bars1 = ax1.bar(methods, speedups, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Speedup = 1.0 (sem melhoria)')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup por Método')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, speedup in zip(bars1, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # 2. Efficiency comparison
    bars2 = ax2.bar(methods, efficiencies, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Eficiência')
    ax2.set_title('Eficiência Paralela')
    ax2.grid(True, alpha=0.3)
    
    for bar, eff in zip(bars2, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Time comparison
    x = np.arange(len(methods))
    width = 0.35
    
    bars3 = ax3.bar(x - width/2, times_serial, width, label='Serial', color='lightcoral', alpha=0.7)
    bars4 = ax3.bar(x + width/2, times_parallel, width, label='Paralelo', color='lightblue', alpha=0.7)
    
    ax3.set_ylabel('Tempo (segundos)')
    ax3.set_title('Tempo de Execução')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Overhead analysis
    overheads = [max(0, t_par - t_ser/mp.cpu_count()) for t_ser, t_par in zip(times_serial, times_parallel)]
    computations = [t_ser/mp.cpu_count() for t_ser in times_serial]
    
    x = np.arange(len(methods))
    bars5 = ax4.bar(x, computations, label='Computação útil', color='green', alpha=0.7)
    bars6 = ax4.bar(x, overheads, bottom=computations, label='Overhead', color='red', alpha=0.7)
    
    ax4.set_ylabel('Tempo (segundos)')
    ax4.set_title('Overhead vs Computação')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_scalability_analysis(strong_results, weak_results=None):
    """
    Plota análise de escalabilidade (strong e weak scaling)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análise de Escalabilidade', fontsize=16, fontweight='bold')
    
    processes = [r['processes'] for r in strong_results]
    speedups = [r['speedup'] for r in strong_results]
    efficiencies = [r['efficiency'] for r in strong_results]
    
    # 1. Strong Scaling - Speedup
    ax1 = axes[0, 0]
    ax1.plot(processes, speedups, 'bo-', linewidth=2, markersize=8, label='Speedup Real')
    ax1.plot(processes, processes, 'r--', linewidth=2, label='Speedup Ideal')
    ax1.set_xlabel('Número de Processos')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Strong Scaling - Speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Strong Scaling - Eficiência
    ax2 = axes[0, 1]
    ax2.plot(processes, efficiencies, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Número de Processos')
    ax2.set_ylabel('Eficiência')
    ax2.set_title('Strong Scaling - Eficiência')
    ax2.grid(True, alpha=0.3)
    
    # 3. Weak Scaling (se disponível)
    if weak_results:
        weak_processes = [r['processes'] for r in weak_results]
        weak_times = [r['time'] for r in weak_results]
        weak_efficiencies = [r['efficiency'] for r in weak_results]
        
        ax3 = axes[1, 0]
        ax3.plot(weak_processes, weak_times, 'mo-', linewidth=2, markersize=8)
        ax3.axhline(y=weak_times[0], color='red', linestyle='--', alpha=0.7, label='Tempo Ideal')
        ax3.set_xlabel('Número de Processos')
        ax3.set_ylabel('Tempo de Execução (s)')
        ax3.set_title('Weak Scaling - Tempo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.plot(weak_processes, weak_efficiencies, 'co-', linewidth=2, markersize=8)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Número de Processos')
        ax4.set_ylabel('Eficiência')
        ax4.set_title('Weak Scaling - Eficiência')
        ax4.grid(True, alpha=0.3)
    else:
        # Gráfico conceitual se não tiver dados weak scaling
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, 'Weak Scaling\n(Problema cresce\ncom recursos)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Weak Scaling - Conceito')
        
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Eficiência ideal\nconstante = 1.0', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Weak Scaling - Meta')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_overhead_breakdown(simple_result, intensive_result):
    """
    Plota breakdown de overhead vs computação
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Overhead vs Computação: Simples vs Intensivo', fontsize=16, fontweight='bold')
    
    methods = ['Simples', 'Intensivo']
    results = [simple_result, intensive_result]
    
    for i, (ax, method, result) in enumerate(zip([ax1, ax2], methods, results)):
        # Calcular componentes
        t_serial = result['time_serial']
        t_parallel = result['time_parallel']
        n_cores = mp.cpu_count()
        
        # Tempo ideal (se fosse perfeitamente paralelo)
        t_ideal = t_serial / n_cores
        
        # Overhead estimado
        overhead = max(0, t_parallel - t_ideal)
        useful_computation = t_ideal
        
        # Pie chart
        if overhead > useful_computation:
            colors = ['red', 'lightcoral']
            labels = [f'Overhead\n{overhead:.3f}s', f'Computação\n{useful_computation:.3f}s']
        else:
            colors = ['green', 'lightgreen'] 
            labels = [f'Computação\n{useful_computation:.3f}s', f'Overhead\n{overhead:.3f}s']
        
        sizes = [useful_computation, overhead]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 10})
        
        # Adicionar informações
        speedup = result['speedup']
        ax.set_title(f'{method}\nSpeedup: {speedup:.2f}x', fontsize=12, fontweight='bold')
        
        # Adicionar explicação
        if speedup > 1.0:
            explanation = "✅ Computação > Overhead"
        else:
            explanation = "❌ Overhead > Computação"
        
        ax.text(0, -1.3, explanation, ha='center', va='center', transform=ax.transAxes, 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_performance_trend(sizes, times_serial, times_parallel, title="Tendência de Performance"):
    """
    Plota tendência de performance com diferentes tamanhos de problema
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Tempos absolutos
    ax1.loglog(sizes, times_serial, 'r-o', label='Serial', linewidth=2, markersize=6)
    ax1.loglog(sizes, times_parallel, 'b-o', label='Paralelo', linewidth=2, markersize=6)
    ax1.set_xlabel('Tamanho do Problema')
    ax1.set_ylabel('Tempo (segundos)')
    ax1.set_title('Tempos de Execução')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup vs tamanho
    speedups = [ts/tp for ts, tp in zip(times_serial, times_parallel)]
    ax2.semilogx(sizes, speedups, 'g-o', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sem speedup')
    ax2.set_xlabel('Tamanho do Problema')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Tamanho do Problema')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_algorithm_comparison_matrix(algorithms_data):
    """
    Matrix de comparação entre diferentes algoritmos
    """
    n_algorithms = len(algorithms_data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Matriz de Comparação de Algoritmos', fontsize=16, fontweight='bold')
    
    names = [alg['name'] for alg in algorithms_data]
    speedups = [alg['speedup'] for alg in algorithms_data]
    efficiencies = [alg['efficiency'] for alg in algorithms_data]
    complexities = [alg.get('complexity', 'N/A') for alg in algorithms_data]
    
    # 1. Speedup heatmap
    ax1 = axes[0, 0]
    speedup_matrix = np.array(speedups).reshape(1, -1)
    im1 = ax1.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(n_algorithms))
    ax1.set_xticklabels(names, rotation=45)
    ax1.set_yticks([])
    ax1.set_title('Speedup Heatmap')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Eficiência vs Speedup scatter
    ax2 = axes[0, 1]
    colors = ['red' if s < 1.0 else 'green' for s in speedups]
    ax2.scatter(speedups, efficiencies, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax2.annotate(name, (speedups[i], efficiencies[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Speedup')
    ax2.set_ylabel('Eficiência')
    ax2.set_title('Eficiência vs Speedup')
    ax2.grid(True, alpha=0.3)
    
    # 3. Complexidade algoritmica
    ax3 = axes[1, 0]
    complexity_counts = {}
    for comp in complexities:
        complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
    
    ax3.pie(complexity_counts.values(), labels=complexity_counts.keys(), autopct='%1.1f%%')
    ax3.set_title('Distribuição de Complexidade')
    
    # 4. Performance summary
    ax4 = axes[1, 1]
    performance_categories = []
    for speedup in speedups:
        if speedup >= 2.0:
            performance_categories.append('Excelente')
        elif speedup >= 1.5:
            performance_categories.append('Bom')
        elif speedup >= 1.0:
            performance_categories.append('Moderado')
        else:
            performance_categories.append('Ruim')
    
    category_counts = {}
    for cat in performance_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    colors_cat = {'Excelente': 'green', 'Bom': 'lightgreen', 'Moderado': 'yellow', 'Ruim': 'red'}
    bars = ax4.bar(category_counts.keys(), category_counts.values(), 
                   color=[colors_cat[cat] for cat in category_counts.keys()])
    ax4.set_ylabel('Número de Algoritmos')
    ax4.set_title('Categorias de Performance')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_educational_infographic():
    """
    Cria infográfico educativo sobre multiprocessing
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Guia Visual: Quando Usar Multiprocessing', fontsize=18, fontweight='bold')
    
    # Remover eixos
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Seção 1: Overhead Domina
    rect1 = Rectangle((0.5, 7), 4, 2.5, facecolor='lightcoral', alpha=0.7, edgecolor='red')
    ax.add_patch(rect1)
    ax.text(2.5, 8.7, 'OVERHEAD DOMINA', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(2.5, 8.3, '• Algoritmos simples', ha='center', va='center', fontsize=11)
    ax.text(2.5, 8.0, '• Poucas operações', ha='center', va='center', fontsize=11)
    ax.text(2.5, 7.7, '• Speedup < 1.0', ha='center', va='center', fontsize=11)
    ax.text(2.5, 7.4, '❌ Não usar multiprocessing', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Seção 2: Computação Domina
    rect2 = Rectangle((5.5, 7), 4, 2.5, facecolor='lightgreen', alpha=0.7, edgecolor='green')
    ax.add_patch(rect2)
    ax.text(7.5, 8.7, 'COMPUTAÇÃO DOMINA', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7.5, 8.3, '• Algoritmos intensivos', ha='center', va='center', fontsize=11)
    ax.text(7.5, 8.0, '• Muitas operações', ha='center', va='center', fontsize=11)
    ax.text(7.5, 7.7, '• Speedup > 1.0', ha='center', va='center', fontsize=11)
    ax.text(7.5, 7.4, '✅ Usar multiprocessing', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Exemplos
    ax.text(5, 6.5, 'EXEMPLOS PRÁTICOS', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Exemplos negativos
    ax.text(2.5, 5.8, 'Speedup Negativo:', ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    ax.text(2.5, 5.4, '• Soma simples de arrays', ha='center', va='center', fontsize=10)
    ax.text(2.5, 5.1, '• Operações básicas', ha='center', va='center', fontsize=10)
    ax.text(2.5, 4.8, '• Poucos cálculos', ha='center', va='center', fontsize=10)
    
    # Exemplos positivos
    ax.text(7.5, 5.8, 'Speedup Positivo:', ha='center', va='center', fontsize=12, fontweight='bold', color='green')
    ax.text(7.5, 5.4, '• Bootstrap estatístico', ha='center', va='center', fontsize=10)
    ax.text(7.5, 5.1, '• Análise modal', ha='center', va='center', fontsize=10)
    ax.text(7.5, 4.8, '• Simulações complexas', ha='center', va='center', fontsize=10)
    
    # Fórmula chave
    ax.text(5, 3.8, 'REGRA DE OURO', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(5, 3.3, 'Tempo_Computação >> Tempo_Overhead', ha='center', va='center', fontsize=14, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Dicas
    ax.text(5, 2.5, 'DICAS PRÁTICAS', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(5, 2.1, '• Teste com problema pequeno primeiro', ha='center', va='center', fontsize=11)
    ax.text(5, 1.8, '• Meça sempre: serial vs paralelo', ha='center', va='center', fontsize=11)
    ax.text(5, 1.5, '• Considere joblib para simplicidade', ha='center', va='center', fontsize=11)
    ax.text(5, 1.2, '• Use Numba para speedups extremos', ha='center', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return fig