# optimization.py

from typing import Dict, List, Set
import copy
from data_structures import *
from analysis import DataLayoutAnalyzer
from cache_info import CacheInfo

class OptimizationPass:
    """Base class for optimization passes."""
    def __init__(self):
        self.changed = False
    
    def run(self, func: BrilFunction) -> BrilFunction:
        """Run the optimization pass on a function."""
        raise NotImplementedError
        
    def did_change(self) -> bool:
        return self.changed

class LoopFusion(OptimizationPass):
    """Fuse adjacent compatible loops."""
    def run(self, func: BrilFunction) -> BrilFunction:
        self.changed = False
        new_instrs = []
        i = 0
        
        while i < len(func.instrs):
            if (i + 1 < len(func.instrs) and
                self._can_fuse_loops(func.instrs[i], func.instrs[i + 1])):
                new_instrs.append(self._fuse_loops(func.instrs[i], func.instrs[i + 1]))
                i += 2
                self.changed = True
                print("Fused adjacent loops")
            else:
                new_instrs.append(func.instrs[i])
                i += 1
        
        func.instrs = new_instrs
        return func
    
    def _can_fuse_loops(self, loop1: Dict, loop2: Dict) -> bool:
        if loop1.get("op") != "loop" or loop2.get("op") != "loop":
            return False
            
        args1 = loop1.get("args", [])
        args2 = loop2.get("args", [])
        
        if len(args1) < 2 or len(args2) < 2:
            return False
            
        return (args1[1] == args2[1] and  
                self._no_dependencies(loop1["body"], loop2["body"]))
    
    def _no_dependencies(self, body1: Dict, body2: Dict) -> bool:
        writes1 = self._get_modified_vars(body1)
        reads2 = self._get_read_vars(body2)
        return not (writes1 & reads2)
    
    def _get_modified_vars(self, body: Dict) -> Set[str]:
        modified = set()
        for instr in body.get("instrs", []):
            if "dest" in instr:
                modified.add(instr["dest"])
            if instr.get("op") == "store":
                modified.add(instr["args"][0])
        return modified
    
    def _get_read_vars(self, body: Dict) -> Set[str]:
        reads = set()
        for instr in body.get("instrs", []):
            if "args" in instr:
                reads.update(arg for arg in instr["args"] 
                           if isinstance(arg, str) and not arg.isdigit())
        return reads
    
    def _fuse_loops(self, loop1: Dict, loop2: Dict) -> Dict:
        fused_body = {
            "instrs": (loop1["body"]["instrs"] + loop2["body"]["instrs"])
        }
        return {
            "op": "loop",
            "args": loop1["args"],
            "body": fused_body
        }

class ArrayPadding(OptimizationPass):
    """Add padding to arrays to avoid cache conflicts."""
    def __init__(self, cache_info: CacheInfo):
        super().__init__()
        self.cache_line_size = cache_info.line_size
    
    def run(self, func: BrilFunction) -> BrilFunction:
        self.changed = False
        new_instrs = []
        
        for instr in func.instrs:
            if instr.get("op") == "alloc":
                padded_instr = self._pad_allocation(instr)
                if padded_instr != instr:
                    print(f"Applied padding to array {instr.get('dest', '')}")
                new_instrs.append(padded_instr)
            else:
                new_instrs.append(instr)
        
        func.instrs = new_instrs
        return func
    
    def _pad_allocation(self, alloc_instr: Dict) -> Dict:
        type_info = alloc_instr.get("type", {})
        if not isinstance(type_info, dict) or "size" not in type_info:
            return alloc_instr
            
        dimensions = type_info["size"]
        if not dimensions:
            return alloc_instr
            
        element_size = 4
        last_dim = dimensions[-1]
        
        elements_per_line = self.cache_line_size // element_size
        padded_dim = ((last_dim + elements_per_line - 1) // 
                     elements_per_line) * elements_per_line
        
        if padded_dim != last_dim:
            self.changed = True
            new_dimensions = dimensions[:-1] + [padded_dim]
            new_type_info = dict(type_info)
            new_type_info["size"] = new_dimensions
            
            return {
                **alloc_instr,
                "type": new_type_info
            }
        
        return alloc_instr

class LayoutOptimizer:
    """Applies data layout optimizations to Bril code."""
    def __init__(self, analyzer: DataLayoutAnalyzer, cache_info: CacheInfo):
        self.analyzer = analyzer
        self.cache_info = cache_info
        self.TILE_SIZE = self._calculate_optimal_tile_size()
        self.passes = [
            LoopFusion(),
            ArrayPadding(cache_info),
            LoopUnrolling(unroll_factor=4), 

        ]
    #TODO: use a better way
    def _calculate_optimal_tile_size(self):
        """Calculate optimal tile size based on cache parameters."""
        try:
            cache_size = self.cache_info.l1_size
            element_size = 4  # assuming 4-byte elements
        
            # Use about 1/3 of L1 cache per tile to leave room for other data
            target_size = cache_size // 3
            elements = target_size // element_size
        
            elements_per_line = self.cache_info.line_size // element_size
        
            # Calculate base tile size as square root of elements
            base_tile_size = int(elements ** 0.5)
            tile_size = (base_tile_size // elements_per_line) * elements_per_line
        
            min_size = max(elements_per_line, 8)  # minimum 8 elements
            max_size = min(256, int((cache_size / element_size) ** 0.5))  # upper limit!
            
            return max(min_size, min(tile_size, max_size))
        except Exception as e:
            print(f"Warning: Error calculating tile size: {e}")
            return 32 
        
    def optimize(self, func: BrilFunction) -> BrilFunction:
        """Main optimization routine."""
        print("\nStarting optimization...")
        patterns = self.analyzer.analyze_function(func)
        print(f"Detected patterns: {patterns}")
        
        optimized = copy.deepcopy(func)
        
        changed = True
        iteration = 0
        while changed and iteration < 10:
            print(f"Optimization iteration {iteration}")
            changed = False
            
            for opt_pass in self.passes:
                optimized = opt_pass.run(optimized)
                if opt_pass.did_change():
                    changed = True
            
            old_instrs = str(optimized.instrs)
            optimized = self._apply_loop_tiling(optimized, patterns)
            optimized = self._apply_loop_interchange(optimized, patterns)
            
            if str(optimized.instrs) != old_instrs:
                changed = True
            
            iteration += 1
        
        return optimized
    
    def _apply_loop_transformations(self, func: BrilFunction,
                                  patterns: Dict[str, ArrayInfo]) -> BrilFunction:
        """Apply loop tiling and interchange transformations."""
        func = self._apply_loop_interchange(func, patterns)
        func = self._apply_loop_tiling(func, patterns)
        return func
    
        
    def _apply_loop_interchange(self, func: BrilFunction, patterns: Dict[str, ArrayInfo]) -> BrilFunction:
        """Perform loop interchange optimization."""
        self.changed = False
    
        def interchange_loops(instrs):
            for i in range(len(instrs)):
                instr = instrs[i]
                if instr.get("op") == "loop":
                    body_instrs = instr.get("body", {}).get("instrs", [])
                    #try to interchange ONCE at each level
                    if body_instrs and body_instrs[0].get("op") == "loop":
                        inner_loop = body_instrs[0]
                        if self._should_interchange(instr, inner_loop, patterns):
                            print(f"Interchanging loops {instr.get('args', [''])[0]} and {inner_loop.get('args', [''])[0]}")
                            # Swap loops
                            instrs[i] = inner_loop
                            inner_loop["body"]["instrs"] = [instr]
                            self._adjust_indices_after_interchange(instr["body"], instr["args"][0], inner_loop["args"][0])
                            self._adjust_indices_after_interchange(inner_loop["body"], inner_loop["args"][0], instr["args"][0])
                            self.changed = True
                            return  
               
                    for nested_instr in body_instrs:
                        if nested_instr.get("op") == "loop":
                            nested_body = nested_instr.get("body", {}).get("instrs", [])
                            interchange_loops(nested_body)
                        
        interchange_loops(func.instrs)
        return func
    
    def _apply_loop_tiling(self, func: BrilFunction, patterns: Dict[str, ArrayInfo]) -> BrilFunction:
        """Apply loop tiling transformation for any nested loop pattern."""
        def should_tile_loops(outer_loop: Dict, inner_loop: Dict) -> bool:
            """Check if nested loops should be tiled based on memory access patterns."""
            outer_body = outer_loop.get("body", {}).get("instrs", [])
            inner_body = inner_loop.get("body", {}).get("instrs", [])
            
            # Check memory accesses
            has_memory_ops = False
            for instr in inner_body:
                if instr.get("op") in ["load", "store"]:
                    has_memory_ops = True
                    array_name = instr.get("args", [""])[0]
                    if array_name in patterns:
                        pattern = patterns[array_name].access_pattern
                        if pattern in [AccessPattern.COLUMN_MAJOR, AccessPattern.STRIDED]:
                            return True
            
            return has_memory_ops
        def tile_nested_loops(instrs: List[Dict], depth: int = 0) -> List[Dict]:
            """Recursively process and tile nested loops."""
            if depth > 5: 
                return instrs
                
            new_instrs = []
            i = 0
            while i < len(instrs):
                instr = instrs[i]
                
                if instr.get("op") == "loop":
                    body_instrs = instr.get("body", {}).get("instrs", [])
                    
                    tiled_body = tile_nested_loops(body_instrs, depth + 1)
                    instr["body"]["instrs"] = tiled_body
                    
                    if (i + 1 < len(instrs) and 
                        instrs[i + 1].get("op") == "loop" and
                        should_tile_loops(instr, instrs[i + 1])):
                        
                        tiled_loops = self._create_tiled_loops(
                            instr, instrs[i + 1])
                        new_instrs.append(tiled_loops)
                        i += 2
                        print(f"Applied tiling to nested loops at depth {depth}")
                    else:
                        new_instrs.append(instr)
                        i += 1
                else:
                    new_instrs.append(instr)
                    i += 1
            
            return new_instrs
        
        func.instrs = tile_nested_loops(func.instrs)
        return func
    
    def _should_interchange(self, loop1: Dict, loop2: Dict,
                          patterns: Dict[str, ArrayInfo]) -> bool:
        """Determine if loops should be interchanged for better locality."""
        if loop1.get("op") != "loop" or loop2.get("op") != "loop":
            return False
        
        body_instrs = loop2.get("body", {}).get("instrs", [])
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                if array_name in patterns:
                    pattern = patterns[array_name].access_pattern
                    if pattern == AccessPattern.COLUMN_MAJOR:
                        return True
        return False
    
    def _should_tile_loop(self, loop: Dict, patterns: Dict[str, ArrayInfo]) -> bool:
        """Determine if loop should be tiled."""
        body_instrs = loop.get("body", {}).get("instrs", [])
        
        has_inner_loop = any(instr.get("op") == "loop" for instr in body_instrs)
        if not has_inner_loop:
            return False
        
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                if array_name in patterns:
                    pattern = patterns[array_name].access_pattern
                    if pattern in [AccessPattern.COLUMN_MAJOR, AccessPattern.STRIDED]:
                        return True
        return False
    
    def _adjust_indices_after_interchange(self, body: Dict, old_var: str, new_var: str):
        """Update array indices after loop interchange."""
        for instr in body.get("instrs", []):
            if instr.get("op") in ["load", "store"]:
                args = instr.get("args", [])
                if len(args) > 1:
                    index_expr = args[1]
                    if isinstance(index_expr, str):
                        args[1] = index_expr.replace(old_var, new_var)
    
    def _create_tiled_loops(self, outer_loop: Dict, inner_loop: Dict) -> Dict:
        """Create tiled version of nested loops."""
        outer_var = outer_loop["args"][0]
        inner_var = inner_loop["args"][0]
        
        def get_bound(args: List[str], default: str = "n") -> str:
            if len(args) >= 3:
                return args[2]
            return default
        
        outer_bound = get_bound(outer_loop["args"])
        inner_bound = get_bound(inner_loop["args"])
        
        # tile loop structure
        #TODO: needs to be FIXED
        return {
            "op": "loop",
            "args": [f"{outer_var}_tile", "0", outer_bound, str(self.TILE_SIZE)],
            "body": {
                "instrs": [{
                    "op": "loop",
                    "args": [f"{inner_var}_tile", "0", inner_bound, str(self.TILE_SIZE)],
                    "body": {
                        "instrs": [{
                            "op": "loop",
                            "args": [
                                outer_var,
                                f"{outer_var}_tile",
                                f"min({outer_var}_tile + {self.TILE_SIZE}, {outer_bound})",
                                "1"
                            ],
                            "body": {
                                "instrs": [{
                                    "op": "loop",
                                    "args": [
                                        inner_var,
                                        f"{inner_var}_tile",
                                        f"min({inner_var}_tile + {self.TILE_SIZE}, {inner_bound})",
                                        "1"
                                    ],
                                    "body": inner_loop["body"]
                                }]
                            }
                        }]
                    }
                }]
            }
        }
        
class LoopUnrolling(OptimizationPass):
    def __init__(self, unroll_factor: int = 4):
        super().__init__()
        self.unroll_factor = unroll_factor
        self.constant_values = {}
        self.MAX_UNROLL_ITERATIONS = 1024 
        
    def run(self, func: BrilFunction) -> BrilFunction:
        self.changed = False
        self.constant_values.clear()
        self._collect_constants(func.instrs)
        new_instrs = []
        
        for instr in func.instrs:
            if self._should_unroll(instr):
                print(f"Unrolling loop with factor {self.unroll_factor}")
                unrolled = self._unroll_loop(instr)
                new_instrs.append(unrolled)
                self.changed = True
            else:
                new_instrs.append(instr)
                if instr.get("op") == "loop":
                    body = instr.get("body", {})
                    body["instrs"] = self.run(BrilFunction("nested", [], None, 
                                            body.get("instrs", []))).instrs
        
        func.instrs = new_instrs
        return func

    def _collect_constants(self, instrs: List[Dict]):
        """Build a map of constant values."""
        for instr in instrs:
            if instr.get("op") == "const":
                dest = instr.get("dest")
                value = instr.get("value")
                if dest is not None and value is not None:
                    self.constant_values[dest] = value
                    print(f"Collected constant: {dest} = {value}")
    
    def _resolve_value(self, arg: str) -> Optional[int]:
        """Resolve a value that might be a constant variable."""
        try:
            return int(arg)
        except ValueError:
            value = self.constant_values.get(arg)
            if value is not None:
                print(f"Resolved constant {arg} to value {value}")
            return value
        
    def _should_unroll(self, instr: Dict) -> bool:
        """Determine if a loop should be unrolled."""
        if instr.get("op") != "loop":
            return False
        
        args = instr.get("args", [])
        if len(args) < 3:
            print(f"Loop has insufficient args: {args}")
            return False
        
        body = instr.get("body", {}).get("instrs", [])
        if not body or any(i.get("op") == "loop" for i in body):
            print("Loop body empty or contains nested loops")
            return False 
            
        start = self._resolve_value(args[1])
        end = self._resolve_value(args[2])
        step = 1 if len(args) <= 3 else self._resolve_value(args[3])
        
        print(f"\nAnalyzing loop for unrolling:")
        print(f"Loop bounds: start={start}, end={end}, step={step}")
        
        if start is None or end is None or step is None:
            print(f"Could not resolve bounds: start={start}, end={end}, step={step}")
            return False
            
        iterations = (end - start) // step
        print(f"Loop has {iterations} iterations")
        
        if iterations < self.unroll_factor:
            print(f"Too few iterations ({iterations}) for unrolling factor {self.unroll_factor}")
            return False
            
        if iterations > self.MAX_UNROLL_ITERATIONS:
            print(f"Too many iterations ({iterations}) for unrolling. Max is {self.MAX_UNROLL_ITERATIONS}")
            return False
        
        print(f"Loop qualifies for unrolling with {iterations} iterations")
        return True

    
    def _unroll_loop(self, loop: Dict) -> Dict:
        """Create unrolled version of the loop."""
        args = loop.get("args", [])
        loop_var = args[0]
        
        start = self._resolve_value(args[1])
        end = self._resolve_value(args[2])
        step = 1 if len(args) <= 3 else self._resolve_value(args[3])
        
        #should never be None since checked in _should_unroll
        assert all(x is not None for x in [start, end, step]), "Loop bounds should be resolved"
        
        new_step = step * self.unroll_factor
        
        original_body = loop.get("body", {}).get("instrs", [])
        unrolled_body = []
        
        def create_temp_var(base: str, index: int) -> str:
            return f"{base}_unroll_{index}"
        
        temp_vars = set()
        
        for i in range(self.unroll_factor):
            offset = i * step
            for instr in original_body:
                new_instr = instr.copy()
                
                if "dest" in new_instr:
                    orig_dest = new_instr["dest"]
                    new_dest = create_temp_var(orig_dest, i)
                    new_instr["dest"] = new_dest
                    temp_vars.add((new_dest, new_instr.get("type")))
                
                if "args" in new_instr:
                    new_args = []
                    for arg in new_instr["args"]:
                        if isinstance(arg, str):
                            if arg == loop_var:
                                new_args.append(f"({loop_var} + {offset})")
                            elif arg.isdigit():
                                new_args.append(arg)
                            else:
                                const_val = self.constant_values.get(arg)
                                if const_val is not None:
                                    new_args.append(str(const_val))
                                else:
                                    for j in range(i + 1):
                                        if arg == create_temp_var(arg, j):
                                            new_args.append(arg)
                                            break
                                    else:
                                        new_args.append(arg)
                        else:
                            new_args.append(arg)
                    new_instr["args"] = new_args
                
                unrolled_body.append(new_instr)
        
        final_body = []
        for var, var_type in temp_vars:
            if var_type:  # Only add const if we have TYPE info
                final_body.append({
                    "op": "const",
                    "dest": var,
                    "type": var_type,
                    "value": 0
                })
        final_body.extend(unrolled_body)
        
        return {
            "op": "loop",
            "args": [loop_var, str(start), str(end), str(new_step)],
            "body": {"instrs": final_body}
        }