# analysis.py

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from data_structures import *

class IndexExpressionParser:
    """Parses array index expressions to determine access patterns."""
    
    def __init__(self):
        self.loop_vars = set()
        
    def parse(self, expr: str) -> Tuple[List[str], int]:
        """
        Parse an index expression to extract loop variables and stride.
        Returns (list of variables, computed stride)
        """
        try:
            terms = expr.replace(' ', '').split('+')
            coefficients = []
            vars_found = []
            
            for term in terms:
                if '*' in term:
                    coef_str, var = term.split('*')
                    try:
                        coef = int(coef_str)
                    except ValueError:
                        coef = 1
                    coefficients.append(coef)
                    vars_found.append(var)
                else:
                    coefficients.append(1)
                    vars_found.append(term)
            
            # Calculate stride based on coefficients
            stride = max(coefficients) if coefficients else 1
            return vars_found, stride
        except Exception as e:
            print(f"Warning: Failed to parse index expression '{expr}': {e}")
            return [], 1

class DataLayoutAnalyzer:
    """Analyzes memory access patterns in Bril code."""
    
    def __init__(self):
        self.array_info: Dict[str, ArrayInfo] = {}
        self.memory_accesses: List[MemoryAccess] = []
        self.loop_nest_info: Dict[int, LoopInfo] = {}
        self.expr_parser = IndexExpressionParser()
    
    def analyze_function(self, func: BrilFunction) -> Dict[str, ArrayInfo]:
        """
        Analyze memory access patterns in a function.
        Returns dictionary mapping array names to their analysis info.
        """
        self._reset_analysis()
        self._collect_array_declarations(func)
        self._analyze_memory_accesses(func.instrs)
        return self._determine_access_patterns()
    
    def _reset_analysis(self):
        """Reset the analyzer state for a new function."""
        self.array_info.clear()
        self.memory_accesses.clear()
        self.loop_nest_info.clear()
    
    def _collect_array_declarations(self, func: BrilFunction):
        """Collect information about array declarations in the function."""
        # Check arguments for array parameters
        for arg in func.args:
            if arg.type.ptr:
                self.array_info[arg.name] = ArrayInfo(
                    dimensions=[],  # Unknown dimensions for parameters
                    access_pattern=AccessPattern.RANDOM,
                    total_accesses=0,
                    stride_pattern=[],
                    element_type=arg.type.base
                )
        
        # Check function body for array allocations
        for instr in func.instrs:
            if instr.get("op") == "alloc":
                name = instr.get("dest")
                type_info = instr.get("type", {})
                dimensions = type_info.get("size", [])
                element_type = type_info.get("element", "int")
                
                self.array_info[name] = ArrayInfo(
                    dimensions=dimensions,
                    access_pattern=AccessPattern.RANDOM,
                    total_accesses=0,
                    stride_pattern=[],
                    element_type=element_type
                )
    
    def _analyze_memory_accesses(self, instrs: List[Dict], loop_depth: int = 0, 
                               current_loop: Optional[LoopInfo] = None):
        """Analyze memory access patterns within a sequence of instructions."""
        for instr in instrs:
            op = instr.get("op")
            
            if op in ["load", "store"]:
                access = self._parse_memory_access(instr, loop_depth, current_loop)
                if access:
                    self.memory_accesses.append(access)
            
            elif op == "loop":
                loop_info = self._parse_loop_info(instr, loop_depth, current_loop)
                self.loop_nest_info[loop_depth] = loop_info
                self._analyze_memory_accesses(
                    loop_info.body.get("instrs", []), 
                    loop_depth + 1, 
                    loop_info
                )
    
    def _parse_memory_access(self, instr: Dict, loop_depth: int, 
                           current_loop: Optional[LoopInfo]) -> Optional[MemoryAccess]:
        """Parse a memory access instruction."""
        args = instr.get("args", [])
        if len(args) < 2:
            return None
            
        var_name = args[0]
        index_expr = str(args[1])
        loop_vars, stride = self.expr_parser.parse(index_expr)
        
        return MemoryAccess(
            variable=var_name,
            index_expr=index_expr,
            line_number=instr.get("pos", {}).get("line", 0),
            loop_depth=loop_depth,
            stride=stride,
            loop_vars=loop_vars
        )
    
    def _parse_loop_info(self, instr: Dict, depth: int, 
                        parent: Optional[LoopInfo]) -> LoopInfo:
        """Parse a loop instruction to extract loop information."""
        args = instr.get("args", [])
        if len(args) < 2:
            raise ValueError(f"Invalid loop instruction: missing arguments: {instr}")
        
        var = args[0]
        end = args[1]
        start = 0  # Default values
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
        """Determine final access patterns for arrays."""
        for access in self.memory_accesses:
            if access.variable in self.array_info:
                array = self.array_info[access.variable]
                array.total_accesses += 1
                if access.stride > 0:
                    array.stride_pattern.append(access.stride)
        
        for info in self.array_info.values():
            info.access_pattern = self._classify_access_pattern(info)
        
        return self.array_info
    
    def _classify_access_pattern(self, info: ArrayInfo) -> AccessPattern:
        """Classify an array's access pattern based on collected information."""
        if not info.stride_pattern:
            return AccessPattern.RANDOM
            
        stride_counts = defaultdict(int)
        for stride in info.stride_pattern:
            stride_counts[stride] += 1
            
        most_common_stride = max(stride_counts.items(), key=lambda x: x[1])[0]
        
        if most_common_stride == 1:
            return AccessPattern.ROW_MAJOR
        elif most_common_stride > 1:
            return AccessPattern.COLUMN_MAJOR
        else:
            return AccessPattern.STRIDED
