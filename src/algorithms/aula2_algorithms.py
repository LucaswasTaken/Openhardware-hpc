"""
Algoritmos para Aula 2 - Paralelismo Avançado e Escalabilidade
Funções otimizadas para demonstrar speedup real em diferentes cenários
"""

import numpy as np
import time
import cProfile
import pstats
import io
from functools import wraps
from joblib import Parallel, delayed
import multiprocessing as mp

# =============================================================================
# Performance Measurement Tools
# =============================================================================

def time_function(func):
    """Decorator para medir tempo de execução"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result
    return wrapper

def profile_function(func, *args, **kwargs):
    """Profile detalhado de uma função"""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    a = np.array(result)
    a = np.sin(0) * np.cos(0)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Top 10 funções
    
    print("📊 Profile detalhado:")
    print(s.getvalue())
    return result

@time_function
def compute_intensive_task(n):
    """Tarefa computacionalmente intensiva para testes"""
    total = 0
    for i in range(n):
        total += i ** 2
        a = np.sin(i) * np.cos(i)
    return total

# =============================================================================
# Statistical Analysis Algorithms (Intensive)
# =============================================================================

def linear_regression_simple(data_chunk):
    """
    Versão SIMPLES - demonstra overhead dominando
    """
    X, y = data_chunk
    
    # Regressão básica rápida
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    beta = np.linalg.solve(XtX, Xty)
    
    # R² básico
    y_pred = np.dot(X, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {'beta': beta, 'r_squared': r_squared}

def linear_regression_intensive_light(data_chunk):
    """
    Worker INTENSIVO LEVE para regressão linear - versão demo mais rápida
    Bootstrap com apenas 100 amostras para demonstração
    """
    X, y = data_chunk
    
    # Regressão básica
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    beta = np.linalg.solve(XtX, Xty)
    
    n, p = X.shape
    
    # BOOTSTRAP LEVE - 100 amostras (vs 500 na versão completa)
    bootstrap_betas = []
    n_bootstrap = 100  # Reduzido para demo mais rápida
    
    np.random.seed(42)  # Reproducibilidade
    for i in range(n_bootstrap):
        # Amostra bootstrap
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Regressão na amostra bootstrap
        XtX_boot = np.dot(X_boot.T, X_boot)
        Xty_boot = np.dot(X_boot.T, y_boot)
        
        # Regularização para estabilidade
        ridge_param = 1e-6
        XtX_boot += ridge_param * np.eye(p)
        
        try:
            beta_boot = np.linalg.solve(XtX_boot, Xty_boot)
            bootstrap_betas.append(beta_boot)
        except np.linalg.LinAlgError:
            continue
    
    bootstrap_betas = np.array(bootstrap_betas)
    
    # Intervalos de confiança
    confidence_intervals = []
    if len(bootstrap_betas) > 0:
        for j in range(p):
            ci_lower = np.percentile(bootstrap_betas[:, j], 2.5)
            ci_upper = np.percentile(bootstrap_betas[:, j], 97.5)
            confidence_intervals.append((ci_lower, ci_upper))
    
    # Calcular R² e estatísticas
    y_pred = np.dot(X, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'beta': beta,
        'r_squared': r_squared,
        'confidence_intervals': confidence_intervals,
        'n_bootstrap': len(bootstrap_betas)
    }


def linear_regression_intensive(data_chunk):
    """
    Versão INTENSIVA - demonstra speedup real
    Análise estatística completa com:
    - Bootstrap para intervalos de confiança
    - Análise de resíduos
    - Estatísticas avançadas
    """
    X, y = data_chunk
    
    # Regressão básica
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    beta = np.linalg.solve(XtX, Xty)
    
    n, p = X.shape
    
    # BOOTSTRAP INTENSIVO - 500 amostras
    bootstrap_betas = []
    n_bootstrap = 500
    
    np.random.seed(42)  # Reproducibilidade
    for i in range(n_bootstrap):
        # Amostra bootstrap
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Regressão na amostra bootstrap
        XtX_boot = np.dot(X_boot.T, X_boot)
        Xty_boot = np.dot(X_boot.T, y_boot)
        
        # Regularização para estabilidade
        ridge_param = 1e-6
        XtX_boot += ridge_param * np.eye(p)
        
        try:
            beta_boot = np.linalg.solve(XtX_boot, Xty_boot)
            bootstrap_betas.append(beta_boot)
        except np.linalg.LinAlgError:
            continue
    
    bootstrap_betas = np.array(bootstrap_betas)
    
    # INTERVALOS DE CONFIANÇA
    confidence_intervals = []
    if len(bootstrap_betas) > 0:
        for j in range(p):
            ci_lower = np.percentile(bootstrap_betas[:, j], 2.5)
            ci_upper = np.percentile(bootstrap_betas[:, j], 97.5)
            confidence_intervals.append((ci_lower, ci_upper))
    
    # ANÁLISE DE RESÍDUOS
    y_pred = np.dot(X, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    residuals = y - y_pred
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': np.mean(((residuals - np.mean(residuals)) / np.std(residuals))**3),
        'kurtosis': np.mean(((residuals - np.mean(residuals)) / np.std(residuals))**4)
    }
    
    return {
        'beta': beta,
        'r_squared': r_squared,
        'confidence_intervals': confidence_intervals,
        'residual_stats': residual_stats,
        'n_bootstrap': len(bootstrap_betas)
    }

def run_regression_analysis_comparison(n_datasets=100, n_points=500, n_features=5, simple=False, light_mode=None):
    """
    Compara regressão serial vs paralela
    light_mode: None (auto), True (light), False (full)
    """
    # Gerar datasets
    datasets = []
    for i in range(n_datasets):
        X = np.random.randn(n_points, n_features)
        X = np.column_stack([np.ones(n_points), X])  # Intercept
        true_beta = np.random.randn(n_features + 1)
        y = np.dot(X, true_beta) + 0.1 * np.random.randn(n_points)
        datasets.append((X, y))
    
    # Escolher função baseada nos parâmetros
    if simple:
        func = linear_regression_simple
        analysis_type = "SIMPLES"
    else:
        # Auto-detectar se deve usar versão light
        if light_mode is None:
            # Use light mode se dataset é pequeno (< 50 datasets ou < 400 pontos)
            use_light = n_datasets < 50 or n_points < 400
        else:
            use_light = light_mode
            
        if use_light:
            func = linear_regression_intensive_light
            analysis_type = "INTENSIVA (Bootstrap 100 amostras - DEMO)"
        else:
            func = linear_regression_intensive
            analysis_type = "INTENSIVA (Bootstrap 500 amostras - COMPLETA)"
    
    print(f"🧮 Regressão {analysis_type}")
    print(f"• {n_datasets} datasets: {n_points} pontos, {n_features} features")
    
    if not simple:
        bootstrap_count = 100 if 'DEMO' in analysis_type else 500
        print(f"• Bootstrap: {bootstrap_count} amostras por regressão")
        print("• Intervalos de confiança e análise de resíduos")
    
    # Serial
    start_time = time.perf_counter()
    results_serial = [func(data) for data in datasets]
    time_serial = time.perf_counter() - start_time
    
    # Paralelo
    start_time = time.perf_counter()
    results_parallel = Parallel(n_jobs=mp.cpu_count())(
        delayed(func)(data) for data in datasets
    )
    time_parallel = time.perf_counter() - start_time
    
    # Métricas
    speedup = time_serial / time_parallel
    efficiency = speedup / mp.cpu_count()
    
    return {
        'analysis_type': analysis_type,
        'n_datasets': n_datasets,
        'time_serial': time_serial,
        'time_parallel': time_parallel,
        'speedup': speedup,
        'efficiency': efficiency,
        'results': results_parallel
    }

# =============================================================================
# Numba Functions (if available)
# =============================================================================

def get_numba_functions():
    """
    Retorna funções Numba se disponível
    """
    try:
        from numba import jit, prange
        
        def matrix_mult_python(A, B):
            """Multiplicação em Python puro"""
            rows_A, cols_A = A.shape
            rows_B, cols_B = B.shape
            C = np.zeros((rows_A, cols_B))
            for i in range(rows_A):
                for j in range(cols_B):
                    for k in range(cols_A):
                        C[i, j] += A[i, k] * B[k, j]
            return C
        
        @jit(nopython=True)
        def matrix_mult_numba_serial(A, B):
            """Multiplicação com Numba serial"""
            rows_A, cols_A = A.shape
            rows_B, cols_B = B.shape
            C = np.zeros((rows_A, cols_B))
            for i in range(rows_A):
                for j in range(cols_B):
                    for k in range(cols_A):
                        C[i, j] += A[i, k] * B[k, j]
            return C
        
        @jit(nopython=True, parallel=True)
        def matrix_mult_numba_parallel(A, B):
            """Multiplicação com Numba paralelo"""
            rows_A, cols_A = A.shape
            rows_B, cols_B = B.shape
            C = np.zeros((rows_A, cols_B))
            for i in prange(rows_A):
                for j in range(cols_B):
                    for k in range(cols_A):
                        C[i, j] += A[i, k] * B[k, j]
            return C
        
        @jit(nopython=True)
        def complex_calculation_serial(n):
            """Cálculo complexo serial"""
            result = 0.0
            for i in range(n):
                result += np.sin(i) * np.cos(i) + np.sqrt(i + 1)
            return result
        
        @jit(nopython=True, parallel=True)
        def complex_calculation_parallel(n):
            """Cálculo complexo paralelo"""
            result = 0.0
            for i in prange(n):
                result += np.sin(i) * np.cos(i) + np.sqrt(i + 1)
            return result
        
        return {
            'available': True,
            'matrix_mult_python': matrix_mult_python,
            'matrix_mult_numba_serial': matrix_mult_numba_serial,
            'matrix_mult_numba_parallel': matrix_mult_numba_parallel,
            'complex_calculation_serial': complex_calculation_serial,
            'complex_calculation_parallel': complex_calculation_parallel
        }
        
    except ImportError:
        return {'available': False}

# =============================================================================
# Utility Functions
# =============================================================================

def format_speedup_result(speedup, efficiency=None):
    """Formata resultado de speedup de forma educativa"""
    if speedup > 1.0:
        improvement = (speedup - 1) * 100
        result = f"✅ Speedup: {speedup:.2f}x ({improvement:.0f}% mais rápido!)"
    else:
        degradation = (1 - speedup) * 100
        result = f"⚠️ Speedup: {speedup:.2f}x ({degradation:.0f}% mais lento - overhead domina)"
    
    if efficiency is not None:
        result += f"\n   Eficiência: {efficiency:.2f} ({efficiency*100:.0f}%)"
    
    return result

def explain_overhead_vs_computation():
    """Explica o conceito de overhead vs computação"""
    return """
💡 CONCEITO CHAVE: Overhead vs Computação

🔹 OVERHEAD de multiprocessing:
  • Criação e destruição de processos
  • Comunicação entre processos (IPC)
  • Serialização/deserialização de dados
  • Sincronização e coordenação

🔹 COMPUTAÇÃO útil:
  • Algoritmos matemáticos intensivos
  • Loops com muitas iterações
  • Operações matriciais complexas
  • Análises estatísticas pesadas

🎯 REGRA DE OURO:
  Speedup positivo APENAS quando: Computação >> Overhead

📊 EXEMPLOS:
  • Cálculo simples: overhead domina → speedup negativo
  • Bootstrap 500x: computação domina → speedup positivo
  • Análise modal: computação domina → speedup positivo
"""