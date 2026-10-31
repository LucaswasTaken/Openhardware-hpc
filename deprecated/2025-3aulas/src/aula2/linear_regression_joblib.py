#!/usr/bin/env python3
"""
Exemplo 7: Regressões lineares paralelas com joblib
Demonstra a simplicidade do joblib para paralelização de loops.
"""

import time
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed

def linear_regression_worker(data_chunk):
    """Worker para regressão linear em chunk de dados"""
    X, y = data_chunk
    
    # Implementação simples de regressão linear: β = (X'X)⁻¹X'y
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    beta = np.linalg.solve(XtX, Xty)
    
    # Calcular R²
    y_pred = np.dot(X, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return beta, r_squared

def generate_regression_data(n_points, n_features, noise=0.1):
    """Gera dados sintéticos para regressão"""
    X = np.random.randn(n_points, n_features)
    X = np.column_stack([np.ones(n_points), X])  # Adicionar intercept
    true_beta = np.random.randn(n_features + 1)
    y = np.dot(X, true_beta) + noise * np.random.randn(n_points)
    return X, y, true_beta

def main():
    print("📊 Regressões Lineares Paralelas com Joblib")
    print("=" * 50)
    
    # Parâmetros
    n_datasets = 1000
    n_points = 1000
    n_features = 5
    n_jobs = mp.cpu_count()
    
    print(f"Executando {n_datasets} regressões lineares")
    print(f"Cada dataset: {n_points} pontos, {n_features} features")
    print(f"Usando {n_jobs} processos\n")
    
    # Gerar datasets
    print("Gerando datasets...")
    datasets = []
    np.random.seed(42)
    
    for i in range(n_datasets):
        X, y, true_beta = generate_regression_data(n_points, n_features)
        datasets.append((X, y))
    
    # Método serial
    print("Executando regressões serial...")
    start_time = time.perf_counter()
    results_serial = [linear_regression_worker(data) for data in datasets]
    time_serial = time.perf_counter() - start_time
    
    # Método paralelo (joblib)
    print("Executando regressões paralelo (joblib)...")
    start_time = time.perf_counter()
    results_parallel = Parallel(n_jobs=n_jobs)(
        delayed(linear_regression_worker)(data) for data in datasets
    )
    time_parallel = time.perf_counter() - start_time
    
    # Análise dos resultados
    speedup = time_serial / time_parallel
    
    # Extrair R² para análise
    r_squared_serial = [r[1] for r in results_serial]
    r_squared_parallel = [r[1] for r in results_parallel]
    
    print("\nResultados:")
    print(f"Tempo serial:   {time_serial:.3f}s")
    print(f"Tempo paralelo: {time_parallel:.3f}s")
    print(f"Speedup:        {speedup:.2f}x")
    
    print(f"\nEstatísticas R²:")
    print(f"R² médio (serial):   {np.mean(r_squared_serial):.4f} ± {np.std(r_squared_serial):.4f}")
    print(f"R² médio (paralelo): {np.mean(r_squared_parallel):.4f} ± {np.std(r_squared_parallel):.4f}")
    
    # Verificar se os resultados são idênticos
    max_diff = max(abs(r_s - r_p) for r_s, r_p in zip(r_squared_serial, r_squared_parallel))
    print(f"Diferença máxima: {max_diff:.2e}")
    print(f"Resultados idênticos: {'✓' if max_diff < 1e-10 else '✗'}")
    
    print("\n💡 Vantagens do Joblib:")
    print("• Interface extremamente simples")
    print("• Otimizações automáticas para arrays NumPy")
    print("• Usado internamente pelo scikit-learn")
    print("• Ideal para loops embaraçosamente paralelos")
    
    print("\n🔧 Aplicações em Engenharia:")
    print("• Análise paramétrica de estruturas")
    print("• Calibração de modelos constitutivos")
    print("• Análise de sensibilidade")
    print("• Processamento de dados experimentais")

if __name__ == "__main__":
    main()