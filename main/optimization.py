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
        ]
    
    def _calculate_optimal_tile_size(self):
        cache_size = self.cache_info.l1_size
        element_size = 8
        block_elements = (cache_size // (3 * element_size))
        tile_size = int(block_elements ** 0.5)
        cache_line_elements = self.cache_info.line_size // element_size
        return max(8, (tile_size // cache_line_elements) * cache_line_elements)
        
    def optimize(self, func: BrilFunction) -> BrilFunction:
        print("Starting optimization...")
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
                    print(f"Pass {opt_pass.__class__.__name__} made changes")
                    changed = True
            
            old_instrs = str(optimized.instrs)
            self.changed = False #reset flag
            optimized = self._apply_loop_interchange(optimized, patterns)
            optimized = self._apply_loop_tiling(optimized, patterns)
            if self.changed:
                print("Loop transformations applied")
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
                    # Only try to interchange once at each level
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
        """Apply loop tiling transformation."""
        self.changed = False
        
        def tile_loops(instrs):
            new_instrs = []
            for instr in instrs:
                if instr.get("op") == "loop":
                    instr["body"]["instrs"] = tile_loops(instr.get("body", {}).get("instrs", []))
                    if self._should_tile_loop(instr, patterns):
                        print(f"Applying tiling to loop {instr.get('args', [''])[0]}")
                        new_instrs.append(self._create_tiled_loop(instr))
                        self.changed = True
                    else:
                        new_instrs.append(instr)
                else:
                    new_instrs.append(instr)
            return new_instrs
        func.instrs = tile_loops(func.instrs)
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
    
    def _create_tiled_loop(self, original_loop: Dict) -> Dict:
        """Create a tiled version of a loop."""
        args = original_loop.get("args", [])
        if len(args) < 2:
            return original_loop
        
        loop_var = args[0]
        end = args[1]
    
        return {
            "op": "loop",
            "args": [f"{loop_var}_tile", "0", end, str(self.TILE_SIZE)],
            "body": {
                "instrs": [
                    {
                        "op": "loop",
                        "args": [
                            loop_var,
                            f"{loop_var}_tile",
                            f"{loop_var}_tile + {self.TILE_SIZE}",  # Changed this line
                            "1"
                        ],
                        "body": original_loop["body"]
                    }
                ]
            }
        }