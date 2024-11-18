# optimization.py

from typing import Dict, List, Set
import copy
from data_structures import *
from analysis import DataLayoutAnalyzer

class LayoutOptimizer:
    """Applies data layout optimizations to Bril code."""
    
    def __init__(self, analyzer: DataLayoutAnalyzer):
        self.analyzer = analyzer
        self.TILE_SIZE = 32 
        
    def optimize(self, func: BrilFunction) -> BrilFunction:
        """
        Apply layout optimizations to a function.
        Returns optimized function.
        """
        patterns = self.analyzer.analyze_function(func)
        optimized_func = copy.deepcopy(func)
        
        self._apply_loop_interchange(optimized_func, patterns)
        self._apply_loop_tiling(optimized_func, patterns)
        
        return optimized_func
    
    def _apply_loop_interchange(self, func: BrilFunction, 
                              patterns: Dict[str, ArrayInfo]):
        """Apply loop interchange optimization for better locality."""
        interchangeable_loops = self._find_interchangeable_loops(func.instrs, patterns)
        
        for loop_pair in interchangeable_loops:
            self._interchange_loop_pair(loop_pair[0], loop_pair[1])
    
    def _find_interchangeable_loops(self, instrs: List[Dict], 
                                  patterns: Dict[str, ArrayInfo]) -> List[tuple]:
        """Find pairs of loops that can be interchanged for better locality."""
        result = []
        
        for i, instr in enumerate(instrs[:-1]):
            if instr.get("op") == "loop":
                next_instr = instrs[i + 1]
                if next_instr.get("op") == "loop":
                    if self._should_interchange(instr, next_instr, patterns):
                        result.append((instr, next_instr))
        
        return result
    
    def _should_interchange(self, loop1: Dict, loop2: Dict, 
                          patterns: Dict[str, ArrayInfo]) -> bool:
        """Determine if interchanging two loops would improve locality."""
        # Simple heuristic: interchange if we detect column-major access
        body_instrs = loop2.get("body", {}).get("instrs", [])
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                if array_name in patterns:
                    if patterns[array_name].access_pattern == AccessPattern.COLUMN_MAJOR:
                        return True
        return False
    
    def _interchange_loop_pair(self, loop1: Dict, loop2: Dict):
        """Perform loop interchange transformation."""
        loop1["args"], loop2["args"] = loop2["args"], loop1["args"]
        
        self._adjust_indices_after_interchange(loop1["body"], loop1["args"][0], 
                                            loop2["args"][0])
        self._adjust_indices_after_interchange(loop2["body"], loop2["args"][0], 
                                            loop1["args"][0])
    
    def _adjust_indices_after_interchange(self, body: Dict, old_var: str, 
                                        new_var: str):
        """Adjust array indices after loop interchange."""
        for instr in body.get("instrs", []):
            if instr.get("op") in ["load", "store"]:
                args = instr.get("args", [])
                if len(args) > 1:
                    index_expr = args[1]
                    if isinstance(index_expr, str):
                        args[1] = index_expr.replace(old_var, new_var)
    
    def _apply_loop_tiling(self, func: BrilFunction, 
                          patterns: Dict[str, ArrayInfo]):
        """Apply loop tiling optimization."""
        for i, instr in enumerate(func.instrs):
            if instr.get("op") == "loop":
                if self._should_tile_loop(instr, patterns):
                    func.instrs[i] = self._create_tiled_loop(instr)
    
    def _should_tile_loop(self, loop: Dict, patterns: Dict[str, ArrayInfo]) -> bool:
        """Determine if a loop should be tiled."""
        body_instrs = loop.get("body", {}).get("instrs", [])
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                if array_name in patterns:
                    pattern = patterns[array_name].access_pattern
                    if pattern in [AccessPattern.COLUMN_MAJOR, AccessPattern.STRIDED]:
                        return True
        return False
    
    def _create_tiled_loop(self, original_loop: Dict) -> Dict:
        """Create a tiled version of a loop."""
        args = original_loop.get("args", [])
        if len(args) < 2:
            return original_loop
            
        loop_var = args[0]
        end = args[1]
        
        tile_loop = {
            "op": "loop",
            "args": [f"{loop_var}_tile", 0, end, self.TILE_SIZE],
            "body": {
                "instrs": [
                    {
                        "op": "loop",
                        "args": [
                            loop_var,
                            f"{loop_var}_tile",
                            f"min({loop_var}_tile + {self.TILE_SIZE}, {end})",
                            1
                        ],
                        "body": original_loop["body"]
                    }
                ]
            }
        }
        
        return tile_loop

class OptimizationPass:
    """Base class for optimization passes."""
    
    def __init__(self):
        self.changed = False
    
    def run(self, func: BrilFunction) -> BrilFunction:
        """Run the optimization pass on a function."""
        raise NotImplementedError
        
    def did_change(self) -> bool:
        """Return whether the pass made any changes."""
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
            else:
                new_instrs.append(func.instrs[i])
                i += 1
        
        func.instrs = new_instrs
        return func
    
    def _can_fuse_loops(self, loop1: Dict, loop2: Dict) -> bool:
        """Check if two loops can be fused."""
        if loop1.get("op") != "loop" or loop2.get("op") != "loop":
            return False
            
        args1 = loop1.get("args", [])
        args2 = loop2.get("args", [])
        
        if len(args1) < 2 or len(args2) < 2:
            return False
            
        return (args1[1] == args2[1] and  
                self._no_dependencies(loop1["body"], loop2["body"]))
    
    def _no_dependencies(self, body1: Dict, body2: Dict) -> bool:
        """Check if there are no dependencies between loop bodies."""
        writes1 = self._get_modified_vars(body1)
        reads2 = self._get_read_vars(body2)
        
        return not (writes1 & reads2)
    
    def _get_modified_vars(self, body: Dict) -> Set[str]:
        """Get variables modified in a loop body."""
        modified = set()
        for instr in body.get("instrs", []):
            if "dest" in instr:
                modified.add(instr["dest"])
            if instr.get("op") == "store":
                modified.add(instr["args"][0])
        return modified
    
    def _get_read_vars(self, body: Dict) -> Set[str]:
        """Get variables read in a loop body."""
        reads = set()
        for instr in body.get("instrs", []):
            if "args" in instr:
                reads.update(arg for arg in instr["args"] 
                           if isinstance(arg, str) and not arg.isdigit())
        return reads
    
    def _fuse_loops(self, loop1: Dict, loop2: Dict) -> Dict:
        """Fuse two compatible loops."""
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
    
    def __init__(self, cache_line_size: int = 64):
        super().__init__()
        self.cache_line_size = cache_line_size
    
    def run(self, func: BrilFunction) -> BrilFunction:
        self.changed = False
        new_instrs = []
        
        for instr in func.instrs:
            if instr.get("op") == "alloc":
                new_instrs.append(self._pad_allocation(instr))
            else:
                new_instrs.append(instr)
        
        func.instrs = new_instrs
        return func
    
    def _pad_allocation(self, alloc_instr: Dict) -> Dict:
        """Add padding to array allocation if beneficial."""
        type_info = alloc_instr.get("type", {})
        if not isinstance(type_info, dict) or "size" not in type_info:
            return alloc_instr
            
        dimensions = type_info["size"]
        if not dimensions:
            return alloc_instr
            
        element_size = 4  # Assume 4 bytes per element
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
    """Main optimizer that combines all optimization passes."""
    
    def __init__(self, analyzer: DataLayoutAnalyzer):
        self.analyzer = analyzer
        self.passes = [
            LoopFusion(),
            ArrayPadding(),
        ]
    
    def optimize(self, func: BrilFunction) -> BrilFunction:
        """Run all optimization passes until convergence."""
        patterns = self.analyzer.analyze_function(func)
        optimized = copy.deepcopy(func)
        
        changed = True
        while changed:
            changed = False
            for opt_pass in self.passes:
                optimized = opt_pass.run(optimized)
                changed |= opt_pass.did_change()
                
            optimized = self._apply_layout_optimizations(optimized, patterns)
        
        return optimized
    
    def _apply_layout_optimizations(self, func: BrilFunction,
                                  patterns: Dict[str, ArrayInfo]) -> BrilFunction:
        """Apply layout-specific optimizations."""
        func = self._apply_loop_transformations(func, patterns)
        
        func = self._apply_data_transformations(func, patterns)
        
        return func
    
    def _apply_loop_transformations(self, func: BrilFunction,
                                  patterns: Dict[str, ArrayInfo]) -> BrilFunction:
        """Apply loop transformations for better locality."""
        # Implement loop interchange and tiling
        for i, instr in enumerate(func.instrs):
            if instr.get("op") == "loop":
                if i + 1 < len(func.instrs) and func.instrs[i + 1].get("op") == "loop":
                    if self._should_interchange(instr, func.instrs[i + 1], patterns):
                        func.instrs[i], func.instrs[i + 1] = (
                            func.instrs[i + 1], func.instrs[i]
                        )
                
                if self._should_tile(instr, patterns):
                    func.instrs[i] = self._create_tiled_loop(instr)
        
        return func
    
    def _apply_data_transformations(self, func: BrilFunction,
                                  patterns: Dict[str, ArrayInfo]) -> BrilFunction:
        """Apply data layout transformations."""

        return func 

    def _should_interchange(self, loop1: Dict, loop2: Dict,
                          patterns: Dict[str, ArrayInfo]) -> bool:
        """Determine if loops should be interchanged."""
        body_instrs = loop2.get("body", {}).get("instrs", [])
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                if array_name in patterns:
                    if patterns[array_name].access_pattern == AccessPattern.COLUMN_MAJOR:
                        return True
        return False
    
    def _should_tile(self, loop: Dict, patterns: Dict[str, ArrayInfo]) -> bool:
        """Determine if loop should be tiled."""
        # Check if loop has intensive array accesses
        body_instrs = loop.get("body", {}).get("instrs", [])
        array_accesses = sum(1 for instr in body_instrs 
                           if instr.get("op") in ["load", "store"])
        return array_accesses > 5  # Arbitrary threshold
    
    def _create_tiled_loop(self, original_loop: Dict) -> Dict:
        """Create a tiled version of a loop."""
        TILE_SIZE = 32  # Could be tuned based on cache size
        
        args = original_loop.get("args", [])
        if len(args) < 2:
            return original_loop
            
        loop_var = args[0]
        end = args[1]
        
        return {
            "op": "loop",
            "args": [f"{loop_var}_tile", 0, end, TILE_SIZE],
            "body": {
                "instrs": [
                    {
                        "op": "loop",
                        "args": [
                            loop_var,
                            f"{loop_var}_tile",
                            f"min({loop_var}_tile + {TILE_SIZE}, {end})",
                            1
                        ],
                        "body": original_loop["body"]
                    }
                ]
            }
        }
