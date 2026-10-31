#!/usr/bin/env python3
"""
Exemplo 6: Integração numérica trapezoidal com futures
Demonstra paralelização de integração numérica usando concurrent.futures.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def function_to_integrate(x):
    """Função exemplo: f(x) = x²·sin(x) + cos(x)"""
    return x**2 * np.sin(x) + np.cos(x)

def trapezoidal_rule_chunk(args):
    """Integração trapezoidal para um chunk do domínio"""
    start, end, n_points, func = args
    
    x = np.linspace(start, end, n_points)
    y = func(x)
    
    # Regra do trapézio
    h = (end - start) / (n_points - 1)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    return integral

def integrate_parallel(func, a, b, n_total, n_processes):
    """Integração paralela usando futures"""
    start_time = time.perf_counter()
    
    # Dividir domínio entre processos
    chunk_width = (b - a) / n_processes
    points_per_chunk = n_total // n_processes
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = []
        
        for i in range(n_processes):
            chunk_start = a + i * chunk_width
            chunk_end = chunk_start + chunk_width
            
            # Último chunk pode ter pontos extras
            if i == n_processes - 1:
                chunk_end = b
                points_per_chunk = n_total - i * points_per_chunk
            
            args = (chunk_start, chunk_end, points_per_chunk, func)
            future = executor.submit(trapezoidal_rule_chunk, args)
            futures.append(future)
        
        # Coletar resultados
        total_integral = sum(future.result() for future in as_completed(futures))
    
    end_time = time.perf_counter()
    return total_integral, end_time - start_time

def integrate_serial(func, a, b, n_points):
    """Integração serial para comparação"""
    start_time = time.perf_counter()
    
    x = np.linspace(a, b, n_points)
    y = func(x)
    
    h = (b - a) / (n_points - 1)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    end_time = time.perf_counter()
    return integral, end_time - start_time

def main():
    print("🧮 Integração Numérica Paralela com Futures")
    print("=" * 50)
    
    # Parâmetros de integração
    a, b = 0, 10  # Domínio de integração
    n_points = 10_000_000  # Número de pontos
    n_processes = mp.cpu_count()
    
    print(f"Integrando f(x) = x²·sin(x) + cos(x) de {a} a {b}")
    print(f"Pontos de integração: {n_points:,}")
    print(f"Processos: {n_processes}\n")
    
    # Comparar serial vs paralelo
    print("Método      Integral        Tempo (s)    Speedup")
    print("-" * 50)
    
    # Serial
    integral_serial, time_serial = integrate_serial(function_to_integrate, a, b, n_points)
    print(f"Serial      {integral_serial:>10.6f}    {time_serial:>8.3f}        -")
    
    # Paralelo
    integral_parallel, time_parallel = integrate_parallel(
        function_to_integrate, a, b, n_points, n_processes
    )
    speedup = time_serial / time_parallel
    print(f"Paralelo    {integral_parallel:>10.6f}    {time_parallel:>8.3f}    {speedup:>6.2f}x")
    
    # Verificar precisão
    error = abs(integral_serial - integral_parallel)
    print(f"\nErro absoluto: {error:.2e}")
    print(f"Precisão: {'✓' if error < 1e-6 else '✗'}")
    
    print("\n💡 Aplicações em Engenharia:")
    print("• Integração de cargas distribuídas")
    print("• Cálculo de momentos e centroides")
    print("• Análise de espectros de resposta")
    print("• Integração de equações diferenciais")

if __name__ == "__main__":
    main()