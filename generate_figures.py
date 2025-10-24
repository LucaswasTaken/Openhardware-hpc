#!/usr/bin/env python3
"""
Generate figures for HPC Python Course presentations
Creates all charts and diagrams for the three classes
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

# Set style for professional plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# Create figures directory
os.makedirs('presentations/figures', exist_ok=True)

def save_figure(fig, filename):
    """Save figure with high DPI for presentations"""
    filepath = f'presentations/figures/{filename}'
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved {filepath}")
    plt.close(fig)

# =====================================
# AULA 1 FIGURES
# =====================================

def create_amdahl_law():
    """Create Amdahl's Law speedup chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    processors = np.array([1, 2, 4, 8, 16, 32, 64])
    
    # Different serial fractions
    s_values = [0.05, 0.10, 0.25, 0.50]
    labels = ['5% serial (95% parallel)', '10% serial (90% parallel)', 
              '25% serial (75% parallel)', '50% serial (50% parallel)']
    
    for s, label in zip(s_values, labels):
        speedup = 1 / (s + (1-s) / processors)
        ax.plot(processors, speedup, 'o-', linewidth=2, markersize=6, label=label)
    
    ax.set_xlabel('Number of Processors', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title("Amdahl's Law - Speedup vs Number of Processors", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(1, 64)
    ax.set_ylim(0, 20)
    
    save_figure(fig, 'amdahl_law.png')

def create_performance_scaling():
    """Create performance scaling comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    processors = [1, 2, 4, 8, 16]
    
    # Strong scaling speedup
    ideal_speedup = processors
    real_90 = [1, 1.8, 3.1, 4.7, 6.4]
    real_75 = [1, 1.6, 2.3, 2.8, 3.2]
    
    ax1.plot(processors, ideal_speedup, 'k--', linewidth=2, label='Ideal Speedup')
    ax1.plot(processors, real_90, 'o-', linewidth=2, markersize=6, label='90% Parallel')
    ax1.plot(processors, real_75, 's-', linewidth=2, markersize=6, label='75% Parallel')
    
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Strong Scaling: Speedup vs Processes')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Strong vs Weak scaling comparison
    problem_sizes = ['Small\n(1K elements)', 'Medium\n(10K elements)', 'Large\n(100K elements)', 'XLarge\n(1M elements)']
    strong_times = [100, 60, 35, 25]  # Time decreases with more processors
    weak_times = [25, 25, 25, 25]    # Time stays constant
    
    x = np.arange(len(problem_sizes))
    width = 0.35
    
    ax2.bar(x - width/2, strong_times, width, label='Strong Scaling\n(Fixed Problem)', alpha=0.8)
    ax2.bar(x + width/2, weak_times, width, label='Weak Scaling\n(Fixed Work/Processor)', alpha=0.8)
    
    ax2.set_xlabel('Problem Configuration')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Strong vs Weak Scaling Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'performance_scaling.png')

def create_threading_vs_multiprocessing():
    """Create architecture comparison diagram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Threading (GIL limited)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    
    # GIL bottleneck
    gil_box = FancyBboxPatch((4, 1), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax1.add_patch(gil_box)
    ax1.text(5, 1.5, 'GIL\nBottleneck', ha='center', va='center', fontweight='bold')
    
    # Threads
    thread_colors = ['#ffeeee', '#ffdddd', '#ffcccc']
    for i, color in enumerate(thread_colors):
        thread_box = FancyBboxPatch((1 + i*0.8, 0.5), 0.6, 0.5, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='gray')
        ax1.add_patch(thread_box)
        ax1.text(1.3 + i*0.8, 0.75, f'T{i+1}', ha='center', va='center')
        # Arrow to GIL
        ax1.arrow(1.6 + i*0.8, 1, 2.4 - i*0.8, 0, head_width=0.1, head_length=0.2, 
                 fc='red', ec='red', alpha=0.7)
    
    # Single CPU core
    cpu_box = FancyBboxPatch((7.5, 1), 1.5, 1, boxstyle="round,pad=0.1",
                            facecolor='#ffcccc', edgecolor='darkred', linewidth=2)
    ax1.add_patch(cpu_box)
    ax1.text(8.25, 1.5, '1 CPU Core\nActive', ha='center', va='center', fontweight='bold')
    
    ax1.arrow(6, 1.5, 1.3, 0, head_width=0.1, head_length=0.2, fc='darkred', ec='darkred')
    
    ax1.set_title('Threading (GIL Limited)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Multiprocessing (True parallelism)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    
    # Processes
    process_colors = ['#ccffcc', '#bbffbb', '#aaffaa', '#99ff99']
    for i, color in enumerate(process_colors):
        # Process box
        proc_box = FancyBboxPatch((0.5 + i*2.2, 0.5), 1.5, 1, boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='green', linewidth=2)
        ax2.add_patch(proc_box)
        ax2.text(1.25 + i*2.2, 1, f'Process {i+1}', ha='center', va='center', fontweight='bold')
        
        # CPU core
        cpu_box = FancyBboxPatch((0.5 + i*2.2, 2), 1.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='darkgreen', linewidth=2)
        ax2.add_patch(cpu_box)
        ax2.text(1.25 + i*2.2, 2.4, f'CPU {i+1}', ha='center', va='center', fontweight='bold')
        
        # Arrow
        ax2.arrow(1.25 + i*2.2, 1.5, 0, 0.4, head_width=0.1, head_length=0.1, 
                 fc='green', ec='green')
    
    ax2.set_title('Multiprocessing (True Parallelism)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'threading_vs_multiprocessing.png')

def create_data_parallelism():
    """Create data parallelism visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Original vectors
    ax.text(5, 7.5, 'Original Data', ha='center', fontsize=14, fontweight='bold')
    
    # Vector A
    for i in range(8):
        rect = Rectangle((1 + i*0.4, 6.5), 0.3, 0.5, facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect)
        ax.text(1.15 + i*0.4, 6.75, f'a{i+1}', ha='center', va='center', fontsize=8)
    ax.text(0.5, 6.75, 'A =', ha='center', va='center', fontweight='bold')
    
    # Vector B  
    for i in range(8):
        rect = Rectangle((1 + i*0.4, 5.8), 0.3, 0.5, facecolor='lightcoral', edgecolor='red')
        ax.add_patch(rect)
        ax.text(1.15 + i*0.4, 6.05, f'b{i+1}', ha='center', va='center', fontsize=8)
    ax.text(0.5, 6.05, 'B =', ha='center', va='center', fontweight='bold')
    
    # Process 1
    ax.text(2.5, 4.5, 'Process 1', ha='center', fontsize=12, fontweight='bold')
    for i in range(4):
        rect = Rectangle((1 + i*0.4, 3.8), 0.3, 0.5, facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect)
        ax.text(1.15 + i*0.4, 4.05, f'a{i+1}', ha='center', va='center', fontsize=8)
    
    for i in range(4):
        rect = Rectangle((1 + i*0.4, 3.1), 0.3, 0.5, facecolor='lightcoral', edgecolor='red')
        ax.add_patch(rect)
        ax.text(1.15 + i*0.4, 3.35, f'b{i+1}', ha='center', va='center', fontsize=8)
    
    for i in range(4):
        rect = Rectangle((1 + i*0.4, 2.4), 0.3, 0.5, facecolor='lightgreen', edgecolor='green')
        ax.add_patch(rect)
        ax.text(1.15 + i*0.4, 2.65, f'c{i+1}', ha='center', va='center', fontsize=8)
    
    # Process 2
    ax.text(7, 4.5, 'Process 2', ha='center', fontsize=12, fontweight='bold')
    for i in range(4):
        rect = Rectangle((5.5 + i*0.4, 3.8), 0.3, 0.5, facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect)
        ax.text(5.65 + i*0.4, 4.05, f'a{i+5}', ha='center', va='center', fontsize=8)
    
    for i in range(4):
        rect = Rectangle((5.5 + i*0.4, 3.1), 0.3, 0.5, facecolor='lightcoral', edgecolor='red')
        ax.add_patch(rect)
        ax.text(5.65 + i*0.4, 3.35, f'b{i+5}', ha='center', va='center', fontsize=8)
    
    for i in range(4):
        rect = Rectangle((5.5 + i*0.4, 2.4), 0.3, 0.5, facecolor='lightgreen', edgecolor='green')
        ax.add_patch(rect)
        ax.text(5.65 + i*0.4, 2.65, f'c{i+5}', ha='center', va='center', fontsize=8)
    
    # Final result
    ax.text(5, 1.5, 'Final Result C', ha='center', fontsize=14, fontweight='bold')
    for i in range(8):
        rect = Rectangle((1 + i*0.4, 0.8), 0.3, 0.5, facecolor='lightgreen', edgecolor='green')
        ax.add_patch(rect)
        ax.text(1.15 + i*0.4, 1.05, f'c{i+1}', ha='center', va='center', fontsize=8)
    ax.text(0.5, 1.05, 'C =', ha='center', va='center', fontweight='bold')
    
    # Arrows
    ax.arrow(2.5, 2.2, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7, 2.2, -2, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_title('Data Parallelism: Vector Addition', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    save_figure(fig, 'data_parallelism.png')

def create_monte_carlo_visualization():
    """Create Monte Carlo method visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Visual representation of Monte Carlo
    circle = plt.Circle((0, 0), 1, fill=False, color='blue', linewidth=2)
    ax1.add_patch(circle)
    ax1.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color='red', linewidth=2))
    
    # Generate sample points
    np.random.seed(42)
    n_points = 1000
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    # Points inside circle
    inside = (x**2 + y**2) <= 1
    ax1.scatter(x[inside], y[inside], c='green', s=1, alpha=0.6, label=f'Inside: {np.sum(inside)}')
    ax1.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.6, label=f'Outside: {np.sum(~inside)}')
    
    pi_estimate = 4 * np.sum(inside) / n_points
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title(f'Monte Carlo π Estimation\nπ ≈ {pi_estimate:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Parallel execution diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    
    # Total samples
    total_box = FancyBboxPatch((3, 6.5), 4, 1, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax2.add_patch(total_box)
    ax2.text(5, 7, 'Total: 1M samples', ha='center', va='center', fontweight='bold')
    
    # Processes
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']
    process_names = ['Process 1\n250k samples', 'Process 2\n250k samples', 
                    'Process 3\n250k samples', 'Process 4\n250k samples']
    
    for i, (color, name) in enumerate(zip(colors, process_names)):
        # Process box
        proc_box = FancyBboxPatch((1 + i*2, 4), 1.8, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='black', linewidth=1)
        ax2.add_patch(proc_box)
        ax2.text(1.9 + i*2, 4.75, name, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrow from total
        ax2.arrow(5, 6.4, -3 + i*2, -0.8, head_width=0.1, head_length=0.1, 
                 fc='blue', ec='blue', alpha=0.7)
        
        # Results
        hits = np.random.randint(196000, 197000)  # Approximate π/4 * 250k
        result_box = FancyBboxPatch((1 + i*2, 2), 1.8, 1, boxstyle="round,pad=0.1",
                                   facecolor='lightyellow', edgecolor='orange', linewidth=1)
        ax2.add_patch(result_box)
        ax2.text(1.9 + i*2, 2.5, f'Hits: {hits}', ha='center', va='center', fontsize=9)
        
        # Arrow to result
        ax2.arrow(1.9 + i*2, 3.9, 0, -0.8, head_width=0.1, head_length=0.1, 
                 fc='black', ec='black')
    
    # Final calculation
    final_box = FancyBboxPatch((3, 0.2), 4, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax2.add_patch(final_box)
    ax2.text(5, 0.6, 'π ≈ 4 × (total_hits / 1M)', ha='center', va='center', fontweight='bold')
    
    # Arrows to final
    for i in range(4):
        ax2.arrow(1.9 + i*2, 1.9, 3 - i*2, -1, head_width=0.1, head_length=0.1, 
                 fc='green', ec='green', alpha=0.7)
    
    ax2.set_title('Parallel Monte Carlo Execution')
    ax2.axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'monte_carlo_visualization.png')

# =====================================
# AULA 2 FIGURES  
# =====================================

def create_joblib_vs_multiprocessing():
    """Compare joblib vs multiprocessing overhead"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    array_sizes = [1, 10, 100, 1000]
    multiprocessing_overhead = [50, 150, 400, 450]
    joblib_overhead = [10, 25, 80, 120]
    
    x = np.arange(len(array_sizes))
    width = 0.35
    
    ax.bar(x - width/2, multiprocessing_overhead, width, label='multiprocessing', alpha=0.8, color='red')
    ax.bar(x + width/2, joblib_overhead, width, label='joblib', alpha=0.8, color='green')
    
    ax.set_xlabel('Array Size (MB)')
    ax.set_ylabel('Overhead (ms)')
    ax.set_title('Overhead Comparison: multiprocessing vs joblib')
    ax.set_xticks(x)
    ax.set_xticklabels(array_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'joblib_vs_multiprocessing.png')

def create_numba_speedup():
    """Create Numba speedup chart by code type"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    code_types = ['Python\nLoops', 'NumPy\nOperations', 'Math\nFunctions', 'I/O\nTasks']
    speedups = [75, 3, 125, 1]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    bars = ax.bar(code_types, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{speedup}x', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Numba Speedup by Code Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 140)
    
    save_figure(fig, 'numba_speedup.png')

def create_scaling_efficiency():
    """Create strong scaling efficiency curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    processes = [1, 2, 4, 8, 16, 32]
    
    # Strong scaling efficiency
    large_problem = [100, 95, 85, 75, 60, 45]
    medium_problem = [100, 90, 70, 50, 30, 20]
    small_problem = [100, 80, 50, 25, 15, 10]
    
    ax1.plot(processes, large_problem, 'o-', linewidth=2, markersize=6, label='Large Problem')
    ax1.plot(processes, medium_problem, 's-', linewidth=2, markersize=6, label='Medium Problem')
    ax1.plot(processes, small_problem, '^-', linewidth=2, markersize=6, label='Small Problem')
    
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Excellent (>80%)')
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Good (>60%)')
    ax1.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Poor (<40%)')
    
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Efficiency (%)')
    ax1.set_title('Strong Scaling Efficiency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Weak scaling time comparison
    ideal_time = [10, 10, 10, 10, 10, 10]
    real_time = [10, 10.2, 10.8, 11.5, 12.5, 14.0]
    
    ax2.plot(processes, ideal_time, '--', linewidth=3, label='Ideal (constant time)', color='green')
    ax2.plot(processes, real_time, 'o-', linewidth=2, markersize=6, label='Real (with overhead)', color='red')
    
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Weak Scaling: Time vs Processes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(8, 16)
    
    plt.tight_layout()
    save_figure(fig, 'scaling_efficiency.png')

def create_optimization_cascade():
    """Create optimization cascade diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    techniques = ['Python\nOriginal', 'NumPy\nVectorized', 'Joblib\nParallel', 'Numba\nCompiled']
    times = [1000, 100, 25, 5]
    speedups = [1, 10, 40, 200]
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3']
    
    # Create cascade bars
    y_positions = np.arange(len(techniques))
    bars = ax.barh(y_positions, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add time and speedup labels
    for i, (bar, time, speedup) in enumerate(zip(bars, times, speedups)):
        width = bar.get_width()
        ax.text(width + 20, bar.get_y() + bar.get_height()/2,
                f'{time}ms\n({speedup}x speedup)', ha='left', va='center', fontweight='bold')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(techniques)
    ax.set_xlabel('Execution Time (ms)')
    ax.set_title('Optimization Cascade: Progressive Performance Improvements', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1200)
    
    # Add arrows showing progression
    for i in range(len(techniques)-1):
        ax.annotate('', xy=(times[i+1] + 50, i+1), xytext=(times[i] - 50, i),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    save_figure(fig, 'optimization_cascade.png')

# =====================================
# AULA 3 FIGURES
# =====================================

def create_cpu_gpu_architecture():
    """Create CPU vs GPU architecture comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # CPU Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # CPU cores (few, complex)
    core_positions = [(2, 7), (7, 7), (2, 3), (7, 3)]
    for i, (x, y) in enumerate(core_positions):
        # Core
        core_box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, boxstyle="round,pad=0.1",
                                 facecolor='#ffcccc', edgecolor='darkred', linewidth=2)
        ax1.add_patch(core_box)
        ax1.text(x, y, f'Core {i+1}\nComplex', ha='center', va='center', fontweight='bold')
        
        # Cache
        cache_box = FancyBboxPatch((x-0.6, y-2.5), 1.2, 0.8, boxstyle="round,pad=0.05",
                                  facecolor='#ffffcc', edgecolor='orange', linewidth=1)
        ax1.add_patch(cache_box)
        ax1.text(x, y-2.1, 'Cache', ha='center', va='center', fontsize=9)
        
        # Arrow to cache
        ax1.arrow(x, y-0.9, 0, -0.7, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Main memory
    memory_box = FancyBboxPatch((1, 0.5), 8, 1, boxstyle="round,pad=0.1",
                               facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax1.add_patch(memory_box)
    ax1.text(5, 1, 'Main Memory (~100 GB/s)', ha='center', va='center', fontweight='bold')
    
    # Arrows to memory
    for x, y in core_positions:
        ax1.arrow(x, y-3.3, 0, -1.5, head_width=0.1, head_length=0.1, 
                 fc='green', ec='green', alpha=0.7)
    
    ax1.set_title('CPU Architecture\n(Latency Optimized)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # GPU Architecture
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Streaming Multiprocessors with many cores
    sm_positions = [(2, 8), (5, 8), (8, 8), (2, 5.5), (5, 5.5), (8, 5.5)]
    for i, (x, y) in enumerate(sm_positions):
        # SM box
        sm_box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, boxstyle="round,pad=0.05",
                               facecolor='#ccffcc', edgecolor='green', linewidth=1)
        ax2.add_patch(sm_box)
        ax2.text(x, y, f'SM {i+1}', ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Many small cores in each SM
        for j in range(8):
            core_x = x - 0.6 + (j % 4) * 0.3
            core_y = y - 0.3 + (j // 4) * 0.3
            core_circle = plt.Circle((core_x, core_y), 0.08, facecolor='lightgreen', 
                                   edgecolor='darkgreen', linewidth=0.5)
            ax2.add_patch(core_circle)
    
    # GPU Memory
    memory_box = FancyBboxPatch((1, 1), 8, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#ccccff', edgecolor='blue', linewidth=2)
    ax2.add_patch(memory_box)
    ax2.text(5, 1.75, 'GPU Memory (~900 GB/s)\nHigh Bandwidth', ha='center', va='center', fontweight='bold')
    
    # Arrows to memory
    for x, y in sm_positions:
        ax2.arrow(x, y-0.7, 0, -2.5, head_width=0.1, head_length=0.1, 
                 fc='blue', ec='blue', alpha=0.7)
    
    ax2.set_title('GPU Architecture\n(Throughput Optimized)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'cpu_gpu_architecture.png')

def create_cuda_hierarchy():
    """Create CUDA execution hierarchy"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # Grid
    grid_box = FancyBboxPatch((1, 1), 10, 8, boxstyle="round,pad=0.2",
                             facecolor='#f0f0f0', edgecolor='black', linewidth=3)
    ax.add_patch(grid_box)
    ax.text(6, 9.2, 'Grid (Entire GPU)', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Blocks
    block_colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']
    block_positions = [(2, 6.5), (6, 6.5), (2, 3.5), (6, 3.5)]
    
    for i, ((x, y), color) in enumerate(zip(block_positions, block_colors)):
        # Block box
        block_box = FancyBboxPatch((x-0.8, y-1), 3.6, 2, boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(block_box)
        ax.text(x+1, y+0.7, f'Block {i}', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Threads in each block (4x4 grid)
        for row in range(4):
            for col in range(4):
                thread_x = x - 0.3 + col * 0.4
                thread_y = y - 0.5 + row * 0.25
                thread_circle = plt.Circle((thread_x, thread_y), 0.08, 
                                         facecolor='white', edgecolor='black', linewidth=1)
                ax.add_patch(thread_circle)
                ax.text(thread_x, thread_y, f'{row*4+col}', ha='center', va='center', fontsize=6)
    
    # Labels
    ax.text(10.5, 6.5, 'Streaming\nMultiprocessor\n(Block)', ha='center', va='center', 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", facecolor='wheat'))
    ax.text(10.5, 3.5, 'CUDA\nCores\n(Threads)', ha='center', va='center', 
           fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    # Arrows
    ax.arrow(9.5, 6.5, -1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(9.5, 3.5, -1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    ax.set_title('CUDA Execution Hierarchy\nGrid → Blocks → Threads', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    save_figure(fig, 'cuda_hierarchy.png')

def create_gpu_memory_hierarchy():
    """Create GPU memory hierarchy"""
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Memory levels (pyramid style)
    levels = [
        ('Registers', '~32KB per SM', '1 cycle', '#00ff00', 8),
        ('Shared Memory', '~48KB per Block', '~20 cycles', '#88ff00', 6),
        ('L1 Cache', '~24KB per SM', '~80 cycles', '#ffff00', 4.5),
        ('L2 Cache', '~6MB total', '~200 cycles', '#ff8800', 3),
        ('Global Memory', '~24GB total', '~400 cycles', '#ff0000', 1.5)
    ]
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    
    y_positions = [10, 8, 6, 4, 2]
    bandwidths = ['8 TB/s', '1.5 TB/s', '~3 TB/s', '~1 TB/s', '900 GB/s']
    
    for i, ((name, size, latency, color, width), y_pos, bandwidth) in enumerate(zip(levels, y_positions, bandwidths)):
        # Memory level box
        x_center = 5
        memory_box = FancyBboxPatch((x_center - width/2, y_pos - 0.6), width, 1.2, 
                                   boxstyle="round,pad=0.1", facecolor=color, 
                                   edgecolor='black', linewidth=2)
        ax.add_patch(memory_box)
        
        # Text
        ax.text(x_center, y_pos, f'{name}\n{size}\n{latency}', 
               ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Bandwidth label
        ax.text(x_center + width/2 + 0.5, y_pos, bandwidth, 
               ha='left', va='center', fontweight='bold', fontsize=9,
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Arrows between levels
        if i < len(levels) - 1:
            ax.arrow(x_center, y_pos - 0.7, 0, -0.6, head_width=0.2, head_length=0.1, 
                    fc='black', ec='black')
    
    # Speed indicators
    ax.text(1, 10, 'FASTEST', ha='center', va='center', fontsize=12, fontweight='bold', 
           rotation=90, color='green')
    ax.text(1, 6, 'SPEED', ha='center', va='center', fontsize=12, fontweight='bold', 
           rotation=90, color='orange')
    ax.text(1, 2, 'SLOWEST', ha='center', va='center', fontsize=12, fontweight='bold', 
           rotation=90, color='red')
    
    ax.set_title('GPU Memory Hierarchy\n(Speed vs Capacity Trade-off)', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    save_figure(fig, 'gpu_memory_hierarchy.png')

def create_gpu_performance_comparison():
    """Create GPU vs CPU performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance scaling with problem size
    grid_sizes = ['128×128', '256×256', '512×512', '1024×1024']
    cpu_times = [0.5, 2.1, 8.7, 35.2]
    cupy_times = [0.05, 0.12, 0.31, 0.89]
    cuda_times = [0.03, 0.08, 0.19, 0.51]
    
    x = np.arange(len(grid_sizes))
    width = 0.25
    
    bars1 = ax1.bar(x - width, cpu_times, width, label='CPU', alpha=0.8, color='red')
    bars2 = ax1.bar(x, cupy_times, width, label='CuPy', alpha=0.8, color='green')
    bars3 = ax1.bar(x + width, cuda_times, width, label='CUDA', alpha=0.8, color='blue')
    
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('CPU vs GPU Performance - Heat Equation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grid_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Speedup comparison
    cupy_speedup = [cpu_time / cupy_time for cpu_time, cupy_time in zip(cpu_times, cupy_times)]
    cuda_speedup = [cpu_time / cuda_time for cpu_time, cuda_time in zip(cpu_times, cuda_times)]
    
    ax2.plot(grid_sizes, cupy_speedup, 'o-', linewidth=3, markersize=8, label='CuPy Speedup', color='green')
    ax2.plot(grid_sizes, cuda_speedup, 's-', linewidth=3, markersize=8, label='CUDA Speedup', color='blue')
    
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Speedup over CPU')
    ax2.set_title('GPU Speedup vs Problem Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add speedup values as text
    for i, (cupy_sp, cuda_sp) in enumerate(zip(cupy_speedup, cuda_speedup)):
        ax2.text(i, cupy_sp + 2, f'{cupy_sp:.0f}x', ha='center', va='bottom', fontweight='bold', color='green')
        ax2.text(i, cuda_sp + 2, f'{cuda_sp:.0f}x', ha='center', va='bottom', fontweight='bold', color='blue')
    
    plt.tight_layout()
    save_figure(fig, 'gpu_performance_comparison.png')

def create_course_evolution():
    """Create complete course performance evolution"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Performance evolution bar chart
    techniques = ['Python\nPure', 'NumPy\nVectorized', 'Multiprocessing\n4 cores', 
                 'Joblib\nOptimized', 'Numba\nCPU JIT', 'CuPy\nGPU', 'Numba\nCUDA']
    speedups = [1, 10, 25, 35, 100, 150, 200]
    colors = ['#ff6b6b', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#01a3a4']
    
    bars = ax1.bar(range(len(techniques)), speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{speedup}x', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Optimization Techniques', fontsize=12)
    ax1.set_ylabel('Speedup Factor', fontsize=12)
    ax1.set_title('HPC Course: Performance Evolution Across 3 Classes', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(techniques)))
    ax1.set_xticklabels(techniques, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 220)
    
    # Class progression roadmap
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6)
    
    # Class boxes
    class_info = [
        ('Class 1\nCPU Parallelism', 2, 4, '#ffcccc', ['Python Serial', 'NumPy', 'Multiprocessing']),
        ('Class 2\nAdvanced Tools', 6, 4, '#ccffcc', ['Joblib', 'Numba JIT', 'Scaling Studies']),
        ('Class 3\nGPU Computing', 10, 4, '#ccccff', ['CuPy', 'Numba CUDA', 'Memory Optimization'])
    ]
    
    for title, x, y, color, topics in class_info:
        # Class box
        class_box = FancyBboxPatch((x-1.5, y-1.5), 3, 3, boxstyle="round,pad=0.2",
                                  facecolor=color, edgecolor='black', linewidth=2)
        ax2.add_patch(class_box)
        ax2.text(x, y+0.8, title, ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Topics
        for i, topic in enumerate(topics):
            ax2.text(x, y+0.2-i*0.3, f'• {topic}', ha='center', va='center', fontsize=9)
    
    # Arrows between classes
    ax2.arrow(3.5, 4, 1, 0, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    ax2.arrow(7.5, 4, 1, 0, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    
    # Performance indicators
    performance_levels = ['1-25x\nSpeedup', '35-100x\nSpeedup', '150-200x+\nSpeedup']
    for i, (perf, (_, x, _, _, _)) in enumerate(zip(performance_levels, class_info)):
        ax2.text(x, 1.5, perf, ha='center', va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
    
    ax2.set_title('Learning Progression: From Serial to Massively Parallel', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'course_evolution.png')

# =====================================
# GENERATE ALL FIGURES
# =====================================

def main():
    """Generate all figures for the course presentations"""
    print("🎨 Generating figures for HPC Python Course presentations...")
    print("=" * 60)
    
    print("\n📊 Aula 1 - CPU Parallelism figures:")
    create_amdahl_law()
    create_performance_scaling()
    create_threading_vs_multiprocessing()
    create_data_parallelism()
    create_monte_carlo_visualization()
    
    print("\n⚙️ Aula 2 - Advanced Parallelism figures:")
    create_joblib_vs_multiprocessing()
    create_numba_speedup()
    create_scaling_efficiency()
    create_optimization_cascade()
    
    print("\n⚡ Aula 3 - GPU Computing figures:")
    create_cpu_gpu_architecture()
    create_cuda_hierarchy()
    create_gpu_memory_hierarchy()
    create_gpu_performance_comparison()
    
    print("\n🎯 Course Summary figures:")
    create_course_evolution()
    
    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print(f"📁 Figures saved in: presentations/figures/")
    print("\n📝 Now updating presentation files to use local images...")

if __name__ == "__main__":
    main()