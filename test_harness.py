import json
import subprocess
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class TestResult:
    name: str
    original_runtime: float
    optimized_runtime: float
    cache_misses_original: int
    cache_misses_optimized: int
    speedup: float

class BrilOptimizationTester:
    def __init__(self, bril_file: str):
        self.bril_file = bril_file
        self.optimizer = DataLayoutOptimizer()
        
    def run_tests(self) -> List[TestResult]:
        results = []
        
        with open(self.bril_file, 'r') as f:
            program = json.load(f)
            
        for func in program['functions']:
            result = self._test_function(func)
            results.append(result)
            
        return results
    
    def _test_function(self, func: Dict) -> TestResult:
        original_stats = self._measure_performance(func)
        
        # Apply optimizations
        access_patterns = self.optimizer.analyze_function(func)
        optimizations = self.optimizer.optimize_layout(access_patterns)
        optimized_func = self.optimizer.generate_optimized_code(func, optimizations)
        
        optimized_stats = self._measure_performance(optimized_func)
        
        return TestResult(
            name=func['name'],
            original_runtime=original_stats['runtime'],
            optimized_runtime=optimized_stats['runtime'],
            cache_misses_original=original_stats['cache_misses'],
            cache_misses_optimized=optimized_stats['cache_misses'],
            speedup=original_stats['runtime'] / optimized_stats['runtime']
        )
    
    def _measure_performance(self, func: Dict) -> Dict:
        """Measure runtime and cache performance"""
        cpp_code = self.optimizer.generate_backend_code(func, target='cpp')
        
        stats = self._run_with_perf(cpp_code)
        
        return stats
    
    def _run_with_perf(self, code: str) -> Dict:
        """Run code with perf for performance measurements"""
        with open('temp.cpp', 'w') as f:
            f.write(code)
            
        subprocess.run(['g++', '-O3', '-fopenmp', 'temp.cpp', '-o', 'temp'])
        
        perf_cmd = [
            'perf', 'stat', '-e', 'cache-misses,cycles',
            './temp'
        ]
        
        start_time = time.time()
        process = subprocess.run(perf_cmd, capture_output=True, text=True)
        end_time = time.time()
        
        cache_misses = self._parse_perf_output(process.stderr)
        
        return {
            'runtime': end_time - start_time,
            'cache_misses': cache_misses
        }
    
    def _parse_perf_output(self, output: str) -> int:
        """Parse cache misses from perf output"""
        for line in output.split('\n'):
            if 'cache-misses' in line:
                return int(line.split()[0].replace(',', ''))
        return 0

def main():
    tester = BrilOptimizationTester('test.bril')
    results = tester.run_tests()
    
    print("\nOptimization Test Results:")
    print("-" * 60)
    for result in results:
        print(f"\nFunction: {result.name}")
        print(f"Original Runtime: {result.original_runtime:.3f}s")
        print(f"Optimized Runtime: {result.optimized_runtime:.3f}s")
        print(f"Speedup: {result.speedup:.2f}x")
        print(f"Cache Misses Reduction: {(result.cache_misses_original - result.cache_misses_optimized) / result.cache_misses_original * 100:.1f}%")

if __name__ == "__main__":
    main()
