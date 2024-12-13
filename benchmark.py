import json
import time
import statistics
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
import subprocess
import sys
from typing import List, Optional

@dataclass
class BenchmarkResult:
    mean_time: float
    std_dev: float
    min_time: float
    max_time: float
    num_runs: int

class ProgramBenchmarker:
    def __init__(self, num_runs: int = 10, timeout: int = 5):
        self.num_runs = num_runs
        self.timeout = timeout
    
    def benchmark_program(self, bril_file: str) -> Optional[BenchmarkResult]:
        """Run performance benchmark on a Bril program."""
        execution_times = []
        
        print(f"Running {self.num_runs} benchmarks for {bril_file}")
        
        for i in range(self.num_runs):
            try:
                start_time = time.perf_counter()
                
                cmd = f"cat {bril_file} | bril2json | brili"
                subprocess.run(cmd, 
                             shell=True,
                             check=True, 
                             capture_output=True,
                             timeout=self.timeout)
                             
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                print(f"Run {i+1}: {execution_time:.6f} seconds")
                
            except subprocess.TimeoutExpired:
                print(f"Run {i+1} timed out after {self.timeout} seconds")
                continue
            except subprocess.CalledProcessError as e:
                print(f"Run {i+1} failed with error: {e}")
                print(f"stderr: {e.stderr.decode()}")
                continue
            except Exception as e:
                print(f"Run {i+1} failed with unexpected error: {e}")
                continue
        
        if not execution_times:
            print(f"All runs failed for {bril_file}")
            return None
            
        return BenchmarkResult(
            mean_time=statistics.mean(execution_times),
            std_dev=statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            min_time=min(execution_times),
            max_time=max(execution_times),
            num_runs=len(execution_times)
        )
    
    def compare_programs(self, input_file: str, output_file: str):
        """Compare performance between original and optimized programs."""
        print("\nStarting benchmark comparison...")
        
        input_results = self.benchmark_program(input_file)
        if not input_results:
            print("Failed to benchmark input program")
            return
            
        output_results = self.benchmark_program(output_file)
        if not output_results:
            print("Failed to benchmark optimized program")
            return
        
        improvement = ((input_results.mean_time - output_results.mean_time) 
                      / input_results.mean_time * 100)
        
        self._print_results(input_results, output_results, improvement)
        self._plot_comparison(input_results, output_results)
    
    def _print_results(self, input_results: BenchmarkResult, 
                      output_results: BenchmarkResult, 
                      improvement: float):
        """Print benchmark results."""
        print("\nBenchmark Results:")
        print("\nOriginal Program:")
        print(f"Mean execution time: {input_results.mean_time:.6f} seconds")
        print(f"Standard deviation: {input_results.std_dev:.6f} seconds")
        print(f"Min time: {input_results.min_time:.6f} seconds")
        print(f"Max time: {input_results.max_time:.6f} seconds")
        
        print("\nOptimized Program:")
        print(f"Mean execution time: {output_results.mean_time:.6f} seconds")
        print(f"Standard deviation: {output_results.std_dev:.6f} seconds")
        print(f"Min time: {output_results.min_time:.6f} seconds")
        print(f"Max time: {output_results.max_time:.6f} seconds")
        
        print(f"\nPerformance Improvement: {improvement:.2f}%")
    
    def _plot_comparison(self, input_results: BenchmarkResult, 
                        output_results: BenchmarkResult):
        """Create visualization of benchmark results."""
        labels = ['Original', 'Optimized']
        means = [input_results.mean_time, output_results.mean_time]
        std_devs = [input_results.std_dev, output_results.std_dev]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, means, yerr=std_devs, capsize=5)
        plt.title('Performance Comparison')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        for i, mean in enumerate(means):
            plt.text(i, mean, f'{mean:.6f}s', 
                    ha='center', va='bottom')
        
        plt.savefig('benchmark_results.png')
        print("\nBenchmark visualization saved as 'benchmark_results.png'")
        plt.close()

def check_dependencies():
    """Check if required tools are installed."""
    tools = ['bril2json', 'brili', 'bril2txt']
    for tool in tools:
        try:
            subprocess.run([tool, '--help'], 
                         capture_output=True, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Error: {tool} not found. Please install bril tools:")
            print("npm install -g bril")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Benchmark Bril programs')
    parser.add_argument('input_file', help='Input Bril file')
    parser.add_argument('output_file', help='Optimized Bril file')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of benchmark runs')
    parser.add_argument('--timeout', type=int, default=5,
                       help='Timeout per run in seconds')
    args = parser.parse_args()
    
    check_dependencies()
    
    benchmarker = ProgramBenchmarker(num_runs=args.runs, timeout=args.timeout)
    benchmarker.compare_programs(args.input_file, args.output_file)

if __name__ == '__main__':
    main()