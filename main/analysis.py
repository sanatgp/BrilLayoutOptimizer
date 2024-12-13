# analysis.py

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
from data_structures import *

@dataclass
class BasicBlock:
    id: int
    instructions: List[Dict]
    predecessors: Set[int]
    successors: Set[int]
    dominators: Set[int]
    phi_nodes: List[Dict]

@dataclass
class SSADefinition:
    var_name: str
    version: int
    defining_instr: Dict
    block_id: int

class CFGBuilder:
    def __init__(self):
        self.blocks: Dict[int, BasicBlock] = {}
        self.current_block_id = 0
        
    def build_cfg(self, instrs: List[Dict]) -> Dict[int, BasicBlock]:
        """Build Control Flow Graph from instructions."""
        self._identify_leaders(instrs)
        self._create_edges()
        self._compute_dominators()
        return self.blocks
    
    def _identify_leaders(self, instrs: List[Dict]):
        """Identify basic block leaders in the instruction stream."""
        self.blocks.clear()
        current_block = None
        
        for i, instr in enumerate(instrs):
            is_leader = (i == 0 or
                        instr.get("op") in ["jmp", "br"] or
                        i > 0 and instrs[i-1].get("op") in ["jmp", "br", "ret"] or
                        "label" in instr)
            
            if is_leader:
                self.current_block_id += 1
                current_block = BasicBlock(
                    id=self.current_block_id,
                    instructions=[],
                    predecessors=set(),
                    successors=set(),
                    dominators=set(),
                    phi_nodes=[]
                )
                self.blocks[self.current_block_id] = current_block
            
            if current_block is not None:
                current_block.instructions.append(instr)
    
    def _create_edges(self):
        """Create edges between basic blocks based on control flow."""
        for block_id, block in self.blocks.items():
            if not block.instructions:
                continue
                
            last_instr = block.instructions[-1]
            next_block_id = block_id + 1
            
            if last_instr.get("op") == "jmp":
                target_label = last_instr.get("labels", [""])[0]
                target_block = self._find_block_by_label(target_label)
                if target_block:
                    block.successors.add(target_block.id)
                    target_block.predecessors.add(block_id)
            
            elif last_instr.get("op") == "br":
                for label in last_instr.get("labels", []):
                    target_block = self._find_block_by_label(label)
                    if target_block:
                        block.successors.add(target_block.id)
                        target_block.predecessors.add(block_id)
            
            elif last_instr.get("op") != "ret" and next_block_id in self.blocks:
                block.successors.add(next_block_id)
                self.blocks[next_block_id].predecessors.add(block_id)
    
    def _find_block_by_label(self, label: str) -> Optional[BasicBlock]:
        """Find basic block containing given label."""
        for block in self.blocks.values():
            for instr in block.instructions:
                if "label" in instr and instr["label"] == label:
                    return block
        return None
    
    def _compute_dominators(self):
        """Compute dominator sets for all basic blocks."""
        all_blocks = set(self.blocks.keys())
        
        # Initialize dominators
        for block in self.blocks.values():
            block.dominators = all_blocks.copy()
        
        entry_block = self.blocks.get(1)
        if entry_block:
            entry_block.dominators = {1}
        
        changed = True
        while changed:
            changed = False
            for block_id, block in self.blocks.items():
                if block_id == 1:
                    continue
                    
                new_doms = {block_id}
                pred_doms = [self.blocks[pred_id].dominators 
                           for pred_id in block.predecessors 
                           if pred_id in self.blocks]
                
                if pred_doms:
                    new_doms.update(set.intersection(*pred_doms))
                    
                if new_doms != block.dominators:
                    block.dominators = new_doms
                    changed = True

class SSAConverter:
    def __init__(self):
        self.var_versions: Dict[str, int] = defaultdict(int)
        self.var_defs: Dict[str, List[SSADefinition]] = defaultdict(list)
        self.current_definitions: Dict[str, SSADefinition] = {}
    
    def convert_to_ssa(self, cfg: Dict[int, BasicBlock]) -> Dict[int, BasicBlock]:
        """Convert CFG to SSA form."""
        # rename variables and add phi nodes
        for block_id, block in cfg.items():
            self._process_block(block)
        
        self._insert_phi_functions(cfg) #insert phi functions
        
        return cfg
    
    def _process_block(self, block: BasicBlock):
        """Process instructions in a block to convert to SSA form."""
        for instr in block.instructions:
            if "dest" in instr:
                var_name = instr["dest"]
                self.var_versions[var_name] += 1
                version = self.var_versions[var_name]
                
                ssa_def = SSADefinition(
                    var_name=var_name,
                    version=version,
                    defining_instr=instr,
                    block_id=block.id
                )
                
                instr["dest"] = f"{var_name}_{version}"
                
                self.var_defs[var_name].append(ssa_def)
                self.current_definitions[var_name] = ssa_def
            
            if "args" in instr:
                new_args = []
                for arg in instr["args"]:
                    if isinstance(arg, str) and arg in self.current_definitions:
                        curr_def = self.current_definitions[arg]
                        new_args.append(f"{arg}_{curr_def.version}")
                    else:
                        new_args.append(arg)
                instr["args"] = new_args
    
    def _insert_phi_functions(self, cfg: Dict[int, BasicBlock]):
        """Insert phi functions at necessary points in the CFG."""
        for block in cfg.values():
            if len(block.predecessors) > 1:
                vars_needing_phi = self._find_vars_needing_phi(block, cfg)
                
                for var in vars_needing_phi:
                    self.var_versions[var] += 1
                    version = self.var_versions[var]
                    
                    phi_node = {
                        "op": "phi",
                        "dest": f"{var}_{version}",
                        "args": [],
                        "labels": []
                    }
                    
                    for pred_id in block.predecessors:
                        pred_block = cfg[pred_id]
                        if var in self.current_definitions:
                            pred_def = self._find_reaching_definition(var, pred_block)
                            if pred_def:
                                phi_node["args"].append(f"{var}_{pred_def.version}")
                                phi_node["labels"].append(str(pred_id))
                    
                    if phi_node["args"]:
                        block.phi_nodes.append(phi_node)
    
    def _find_vars_needing_phi(self, block: BasicBlock, cfg: Dict[int, BasicBlock]) -> Set[str]:
        """Find variables that need phi functions at the start of a block."""
        vars_needing_phi = set()
        
        for pred_id in block.predecessors:
            pred_block = cfg[pred_id]
            for instr in pred_block.instructions:
                if "dest" in instr:
                    base_var = instr["dest"].split('_')[0]
                    vars_needing_phi.add(base_var)
        
        return vars_needing_phi
    
    def _find_reaching_definition(self, var: str, block: BasicBlock) -> Optional[SSADefinition]:
        """Find the reaching definition of a variable at the end of a block."""
        for instr in reversed(block.instructions):
            if "dest" in instr and instr["dest"].split('_')[0] == var:
                version = int(instr["dest"].split('_')[1])
                return SSADefinition(var, version, instr, block.id)
        return self.current_definitions.get(var)

class IndexExpressionParser:
    def parse(self, expr: str) -> Tuple[List[str], Dict[str, int]]:
        """Parse array index expressions without assumptions about array sizes."""
        try:
            if not isinstance(expr, str):
                return [], {}

            expr = expr.replace(' ', '')
            terms = expr.split('+')
            var_strides = {}
            symbolic_terms = {}

            for term in terms:
                factors = term.split('*')
                stride = 1
                var = None
                has_symbolic = False
            
                for factor in factors:
                    if factor.isdigit():
                        stride *= int(factor)
                    elif factor.isalpha():
                        if var is None:
                            var = factor
                        else:
                            has_symbolic = True
                            symbolic_terms[factor] = stride
            
                if var is not None:
                    if has_symbolic:
                        var_strides[var] = symbolic_terms
                    else:
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
        self.cfg_builder = CFGBuilder()
        self.ssa_converter = SSAConverter()
    
    def analyze_function(self, func: BrilFunction) -> Dict[str, ArrayInfo]:
        print("\nStarting memory access pattern analysis...")
        self._reset_analysis()
        
        #  build CFG and convert to SSA
        cfg = self.cfg_builder.build_cfg(func.instrs)
        print(f"Built CFG with {len(cfg)} basic blocks")
        ssa_cfg = self.ssa_converter.convert_to_ssa(cfg)
        print("Converted to SSA form")
        
        self._collect_array_declarations(func)
        
        self._analyze_memory_accesses_ssa(ssa_cfg)
        
        self._analyze_memory_accesses(func.instrs)
        
        patterns = self._determine_access_patterns()
        
        self._print_analysis_results(patterns)
        
        return patterns
    
    def _reset_analysis(self):
        self.array_info.clear()
        self.memory_accesses.clear()
        self.loop_nest_info.clear()
    
    def _analyze_memory_accesses_ssa(self, cfg: Dict[int, BasicBlock]):
        """Analyze memory accesses using SSA form for more precise analysis."""
        for block in cfg.values():
            loop_depth = self._calculate_loop_depth(block, cfg)
            
            #phi nodes first
            for phi in block.phi_nodes:
                if phi["dest"].split('_')[0] in self.array_info:
                    self._process_phi_node(phi, block.id, loop_depth)
            
            for instr in block.instructions:
                if instr.get("op") in ["load", "store"]:
                    self._process_memory_access(instr, block.id, loop_depth)
    def _analyze_memory_accesses(self, instrs: List[Dict], loop_depth: int = 0, 
                               current_loop: Optional[LoopInfo] = None):
        """Original loop analysis implementation"""
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
        """Enhanced loop tiling detection"""
        body_instrs = loop.get("body", {}).get("instrs", [])
    
        # nested loops
        has_inner_loop = any(instr.get("op") == "loop" for instr in body_instrs)
        if not has_inner_loop:
            return False
    
        for instr in body_instrs:
            if instr.get("op") in ["load", "store"]:
                array_name = instr.get("args", [""])[0]
                index_expr = str(instr.get("args", ["", ""])[1])
                
                #column-major access patterns
                if "k*n" in index_expr or "n*k" in index_expr:
                    print(f"Found column-major access in loop: {index_expr}")
                    return True
                
                #array access pattern
                if array_name in patterns:
                    pattern = patterns[array_name].access_pattern
                    if pattern in [AccessPattern.COLUMN_MAJOR, AccessPattern.STRIDED]:
                        print(f"Found {pattern} access pattern in array {array_name}")
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

    def _calculate_loop_depth(self, block: BasicBlock, cfg: Dict[int, BasicBlock]) -> int:
        """Calculate loop depth for a basic block."""
        depth = 0
        current_block = block
        visited = set()
        
        while current_block and current_block.id not in visited:
            visited.add(current_block.id)
            back_edges = [pred for pred in current_block.predecessors 
                         if pred in current_block.dominators]
            depth += len(back_edges)
            
            # Move to immediate dominator
            dominator_id = min(current_block.dominators - {current_block.id}, default=None)
            current_block = cfg.get(dominator_id) if dominator_id else None
        
        return depth
    
    def _process_phi_node(self, phi: Dict, block_id: int, loop_depth: int):
        """Process phi node for array access pattern analysis."""
        var_base = phi["dest"].split('_')[0]
        if var_base in self.array_info:
            self.array_info[var_base].total_accesses += 1
            
            prev_version = None
            for arg in phi["args"]:
                curr_version = int(arg.split('_')[1])
                if prev_version is not None:
                    stride = abs(curr_version - prev_version)
                    self.array_info[var_base].stride_pattern.append(stride)
                prev_version = curr_version
    
    def _process_memory_access(self, instr: Dict, block_id: int, loop_depth: int):
        """Process memory access instruction with SSA information."""
        args = instr.get("args", [])
        if len(args) >= 2:
            array_name = args[0].split('_')[0] 
            index_expr = str(args[1])
            
            if array_name in self.array_info:
                loop_vars, var_strides = self.expr_parser.parse(index_expr)
                
                access = MemoryAccess(
                    variable=array_name,
                    index_expr=index_expr,
                    line_number=instr.get("pos", {}).get("line", 0),
                    loop_depth=loop_depth,
                    stride=max(var_strides.values(), default=0),
                    loop_vars=loop_vars
                )
                
                self.memory_accesses.append(access)
                self.array_info[array_name].total_accesses += 1
                self.array_info[array_name].stride_pattern.append(access.stride)
    
    def _print_analysis_results(self, patterns: Dict[str, ArrayInfo]):
        """Print detailed analysis results."""
        print("\nAnalysis Results:")
        for name, info in patterns.items():
            print(f"\nArray: {name}")
            print(f"  Dimensions: {info.dimensions}")
            print(f"  Total accesses: {info.total_accesses}")
            print(f"  Access pattern: {info.access_pattern}")
            print(f"  Stride pattern: {info.stride_pattern}")
            print(f"  Element type: {info.element_type}")
            if info.access_pattern == AccessPattern.RANDOM:
                print("  Warning: Random access pattern detected")
            elif info.access_pattern == AccessPattern.COLUMN_MAJOR:
                print("  Note: Column-major access pattern may benefit from layout optimization")

    def _collect_array_declarations(self, func: BrilFunction):
        """Collect array declarations and arguments with dynamic sizing."""
        for arg in func.args:
            if arg.type.ptr:
                dimensions = self._extract_dimensions(arg.type)
                self.array_info[arg.name] = ArrayInfo(
                    dimensions=dimensions,
                    access_pattern=AccessPattern.RANDOM,
                   total_accesses=0,
                    stride_pattern=[],
                    element_type=arg.type.base
                )
                print(f"Registered array argument: {arg.name} with dims {dimensions}")
    
        for instr in func.instrs:
            if instr.get("op") == "alloc":
                name = instr.get("dest")
                type_info = instr.get("type", {})

                dimensions = self._extract_dimensions(type_info)
                element_type = type_info.get("element", "int")
            
                self.array_info[name] = ArrayInfo(
                    dimensions=dimensions,
                    access_pattern=AccessPattern.RANDOM,
                   total_accesses=0,
                    stride_pattern=[],
                    element_type=element_type
                )
                print(f"Registered allocated array: {name} with dimensions {dimensions}")

    def _extract_dimensions(self, type_info: Union[Dict, BrilType]) -> List[int]:
        """Extract array dimensions from type information."""
        try:
            if isinstance(type_info, dict):
                if "size" in type_info:
                    size = type_info["size"]
                    # return a list of integers
                    if isinstance(size, list):
                        return [int(dim) for dim in size]
                    return [int(size)]
                elif "ptr" in type_info:
                    return []
            elif isinstance(type_info, BrilType):
                if hasattr(type_info, 'params') and type_info.params:
                    return [int(p) for p in type_info.params if str(p).isdigit()]
            return []
        except Exception as e:
            print(f"Warning: Error extracting dimensions: {e}")
            return []

    def _determine_access_patterns(self) -> Dict[str, ArrayInfo]:
        """Enhanced access pattern analysis for matrix operations."""
        print("\nDetermining access patterns...")

        for name, info in self.array_info.items():
            if not info.stride_pattern:
                continue

            try:
                matrix_dim = int(info.dimensions[0]) if info.dimensions else 1024
                inner_dim = int(info.dimensions[1]) if len(info.dimensions) >= 2 else 1024
            except (IndexError, ValueError):
                matrix_dim = 1024
                inner_dim = 1024
        
            strides = info.stride_pattern
            n_accesses = len(strides)
            if n_accesses == 0:
                continue
    
            def compare_stride(stride, value):
                if isinstance(stride, dict):
                    return any(v == value for v in stride.values())
                try:
                    return int(stride) == value
                except (TypeError, ValueError):
                    return False
        
            def is_large_stride(stride):
                if isinstance(stride, dict):
                    return any(v > matrix_dim for v in stride.values())
                try:
                    return int(stride) > matrix_dim
                except (TypeError, ValueError):
                    return False
    
            row_major_strides = sum(1 for s in strides if compare_stride(s, 1))
            col_major_strides = sum(1 for s in strides if compare_stride(s, matrix_dim))
            large_strides = sum(1 for s in strides if is_large_stride(s))
    
        # ratios
            row_ratio = row_major_strides / n_accesses if n_accesses > 0 else 0
            col_ratio = col_major_strides / n_accesses if n_accesses > 0 else 0
            large_ratio = large_strides / n_accesses if n_accesses > 0 else 0
    
            if col_ratio >= 0.3 or large_ratio >= 0.3:
                info.access_pattern = AccessPattern.COLUMN_MAJOR
                print(f"Array {name}: Column-major access (ratio: {col_ratio:.2f})")
            elif row_ratio >= 0.3:
                info.access_pattern = AccessPattern.ROW_MAJOR
                print(f"Array {name}: Row-major access (ratio: {row_ratio:.2f})")
            else:
                info.access_pattern = AccessPattern.STRIDED
                print(f"Array {name}: Strided access pattern")
    
        return self.array_info