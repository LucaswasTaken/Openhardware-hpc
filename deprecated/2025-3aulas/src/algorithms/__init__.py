"""
Algorithms package for HPC course
Provides educational implementations of parallel computing examples
"""

# Make imports available at package level
from .vector_operations import (
    vector_sum_serial,
    vector_sum_numpy,
    vector_sum_threading,
    run_vector_sum_comparison,
    demonstrate_gil_limitation
)

from .monte_carlo import (
    monte_carlo_pi_serial, 
    monte_carlo_pi_threading,
    run_monte_carlo_comparison,
    run_multiprocessing_demo as run_monte_carlo_multiprocessing_demo
)

from .matrix_operations import (
    matrix_multiply_serial,
    matrix_multiply_threading, 
    run_matrix_comparison,
    run_matrix_multiprocessing_demo
)

from .structural_analysis import (
    structural_analysis_serial,
    structural_analysis_vectorized,
    structural_analysis_threading,
    run_structural_comparison,
    plot_structural_results,
    run_structural_multiprocessing_demo
)

__all__ = [
    'vector_sum_serial',
    'vector_sum_numpy',
    'vector_sum_threading',
    'run_vector_sum_comparison',
    'demonstrate_gil_limitation',
    'monte_carlo_pi_serial',
    'monte_carlo_pi_threading', 
    'run_monte_carlo_comparison',
    'run_monte_carlo_multiprocessing_demo',
    'matrix_multiply_serial',
    'matrix_multiply_threading',
    'run_matrix_comparison', 
    'run_matrix_multiprocessing_demo',
    'structural_analysis_serial',
    'structural_analysis_vectorized', 
    'structural_analysis_threading',
    'run_structural_comparison',
    'plot_structural_results',
    'run_structural_multiprocessing_demo'
]