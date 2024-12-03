# analysis.py

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from data_structures import *

class IndexExpressionParser:
    def parse(self, expr: str) -> Tuple[List[str], Dict[str, int]]:
        try:
            if not isinstance(expr, str):
                return [], {}

            expr = expr.replace(' ', '')
            terms = expr.split('+')
            var_strides = {}

            for term in terms:
                factors = term.split('*')
                stride = 1
                var = None
                for factor in factors:
                    if factor.isdigit():
                        stride *= int(factor)
                    elif factor == 'n':
                        stride *= 1024  # Lets assume n = 1024 for simplicity
                    else:
                        var = factor
                if var is not None:
                    var_strides[var] = stride

            return list(var_strides.keys()), var_strides

        except Exception as e:
            print(f"Warning: Failed to parse index expression '{expr}': {e}")
            return [], {}


class DataLayoutAnalyzer:
    def __init__(self):
        self.array_info: Dict[str, ArrayInfo] = {}
        self.memory_accesses: List[MemoryAccess] = []
        self.loop_nest_info: Dict[int, LoopInfo] = {}
        self.expr_parser = IndexExpressionParser()
    
    def analyze_function(self, func: BrilFunction) -> Dict[str, ArrayInfo]:
        print("\nStarting memory access pattern analysis...")
        self._reset_analysis()
        self._collect_array_declarations(func)
        self._analyze_memory_accesses(func.instrs)
        patterns = self._determine_access_patterns()
        
        for name, info in patterns.items():
            print(f"\nArray {name}:")
            print(f"  Dimensions: {info.dimensions}")
            print(f"  Total accesses: {info.total_accesses}")
            print(f"  Stride pattern: {info.stride_pattern}")
            print(f"  Access pattern: {info.access_pattern}")
            if info.access_pattern == AccessPattern.RANDOM:
                print(f"  Warning: Array {name} was detected as RANDOM access pattern")
        
        return patterns
    
    def _reset_analysis(self):
        self.array_info.clear()
        self.memory_accesses.clear()
        self.loop_nest_info.clear()
    
    def _collect_array_declarations(self, func: BrilFunction):
        for arg in func.args:
            if arg.type.ptr:
                self.array_info[arg.name] = ArrayInfo(
                    dimensions=[1024, 1024],
                    access_pattern=AccessPattern.RANDOM,
                    total_accesses=0,
                    stride_pattern=[],
                    element_type=arg.type.base
                )
                print(f"Registered array argument: {arg.name}")
        
        for instr in func.instrs:
            if instr.get("op") == "alloc":
                name = instr.get("dest")
                type_info = instr.get("type", {})
                dimensions = type_info.get("size", [1024, 1024])
                element_type = type_info.get("element", "int")
                
                self.array_info[name] = ArrayInfo(
                    dimensions=dimensions,
                    access_pattern=AccessPattern.RANDOM,
                    total_accesses=0,
                    stride_pattern=[],
                    element_type=element_type
                )
                print(f"Registered allocated array: {name} with dimensions {dimensions}")
    
    def _analyze_memory_accesses(self, instrs: List[Dict], loop_depth: int = 0, 
                               current_loop: Optional[LoopInfo] = None):
        for instr in instrs:
            op = instr.get("op")
            
            if op in ["load", "store"]:
                args = instr.get("args", [])
                if len(args) >= 2:
                    array_name = args[0]
                    index_expr = str(args[1])
                    print(f"Analyzing array access: {array_name}[{index_expr}]")

                    loop_vars, var_strides = self.expr_parser.parse(index_expr)
                    access = MemoryAccess(
                        variable=array_name,
                        index_expr=index_expr,
                        line_number=instr.get("pos", {}).get("line", 0),
                        loop_depth=loop_depth,
                        stride=0, 
                        loop_vars=loop_vars
                    )

                    if current_loop:
                        innermost_var = current_loop.var
                        access.stride = var_strides.get(innermost_var, 0)
                    else:
                        access.stride = 0 

                    if array_name in self.array_info:
                        self.memory_accesses.append(access)
                        self.array_info[array_name].total_accesses += 1
                        self.array_info[array_name].stride_pattern.append(access.stride)
                        print(f"Recorded {array_name} access with stride {access.stride}")
            
            elif op == "loop":
                loop_info = self._parse_loop_info(instr, loop_depth, current_loop)
                self.loop_nest_info[loop_depth] = loop_info
                if "body" in instr:
                    self._analyze_memory_accesses(
                        instr["body"].get("instrs", []),
                        loop_depth + 1,
                        loop_info
                    )
 

    def _should_tile_loop(self, loop: Dict, patterns: Dict[str, ArrayInfo]) -> bool:
        body_instrs = loop.get("body", {}).get("instrs", [])
    
        #nested loops
        has_inner_loop = any(instr.get("op") == "loop" for instr in body_instrs)
        if not has_inner_loop:
            return False
    
        #array accesses
        array_patterns = []
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                index_expr = str(instr.get("args", ["", ""])[1])
                if "k*n" in index_expr or "n*k" in index_expr:
                    print(f"Found column-major access in loop: {index_expr}")
                    return True
    
        return False
    
    def _parse_loop_info(self, instr: Dict, depth: int, 
                        parent: Optional[LoopInfo]) -> LoopInfo:
        args = instr.get("args", [])
        if len(args) < 2:
            raise ValueError(f"Invalid loop instruction: missing arguments: {instr}")
        
        var = args[0]
        end = args[1]
        start = 0
        step = 1
        
        if len(args) > 2:
            start = args[1]
            end = args[2]
        if len(args) > 3:
            step = args[3]
            
        return LoopInfo(
            var=var,
            start=start,
            end=end,
            step=step,
            body=instr.get("body", {}),
            depth=depth,
            parent=parent
        )
    
    def _determine_access_patterns(self) -> Dict[str, ArrayInfo]:
        print("\nDetermining access patterns...")
        
        for access in self.memory_accesses:
            if access.variable in self.array_info:
                array = self.array_info[access.variable]
                array.total_accesses += 1
                array.stride_pattern.append(access.stride)
                print(f"Access to {access.variable}: stride {access.stride}")
        
        for name, info in self.array_info.items():
            prev_pattern = info.access_pattern
            info.access_pattern = self._classify_access_pattern(info)
            print(f"\nArray {name} pattern analysis:")
            print(f"  Previous pattern: {prev_pattern}")
            print(f"  New pattern: {info.access_pattern}")
            print(f"  Based on strides: {info.stride_pattern}")
        
        return self.array_info
    
    def _classify_access_pattern(self, info: ArrayInfo) -> AccessPattern:
        if not info.stride_pattern:
            return AccessPattern.RANDOM

        n_accesses = len(info.stride_pattern)
        if n_accesses == 0:
            return AccessPattern.RANDOM

        matrix_dim = info.dimensions[0] if info.dimensions else 1024
        large_strides = sum(1 for s in info.stride_pattern if s >= matrix_dim / 2)
        unit_strides = sum(1 for s in info.stride_pattern if s == 1)

        if large_strides / n_accesses >= 0.3:
            return AccessPattern.COLUMN_MAJOR
        elif unit_strides / n_accesses >= 0.7:
            return AccessPattern.ROW_MAJOR
        else:
            return AccessPattern.STRIDED