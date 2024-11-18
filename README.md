# BrilLayoutOptimizer
Cache-aware optimizations for Bril, enhancing performance with loop transformations and memory access analysis.

## Usage

Run the optimizer on a Bril JSON file:

```bash
python main.py input.json -o output.json --cache-size  -v
```


### Choosing Cache Size

Specify the cache size based on your system:

* **L1 Cache**: Use for smaller data sets (e.g., `--cache-size 32768` for 32 KB)
* **L2 Cache**: For larger data sets (e.g., `--cache-size 262144` for 256 KB)
* **L3 Cache**: For very large data or server workloads (e.g., `--cache-size 8388608` for 8 MB)
 


Check cache sizes:

```bash
# Linux
lscpu | grep 'cache'

# macOS
sysctl -a | grep 'cachesize'
```


## Running the Optimizer

```bash
# Run with typical L1 cache size
python main.py input.json -o output.json --cache-size 32768 -v

# Verify optimizations
python optimization_checker.py
```

## Testing

Run the test suite:

```bash
pytest tests/
```
