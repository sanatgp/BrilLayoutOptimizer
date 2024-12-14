# analysis.py

from typing import Dict, List, Set, Optional, NamedTuple, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from data_structures import *


@dataclass
class SSADefinition:
    var_name: str
    version: int
    defining_instr: Dict
    block_id: int

@dataclass
class DominanceInfo:
    idom: Optional[int] = None
    dom_frontier: Set[int] = field(default_factory=set)
    dominators: Set[int] = field(default_factory=set)

@dataclass
class BasicBlock:
    id: int
    instructions: List[Dict] = field(default_factory=list)
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    phi_nodes: List[Dict] = field(default_factory=list)
    dominance: DominanceInfo = field(default_factory=DominanceInfo)


class Edge(NamedTuple):
    source: int
    target: int
    edge_type: str

class CFGBuilder:
    def __init__(self):
        self.blocks: Dict[int, BasicBlock] = {}
        self.current_block_id = 0
        self.edges: List[Edge] = []
        self.entry_block_id: Optional[int] = None
        self.exit_block_id: Optional[int] = None
        
    def build_cfg(self, instrs: List[Dict]) -> Dict[int, BasicBlock]:
        """Build Control Flow Graph with enhanced analysis."""
        self._reset()
        self._identify_leaders(instrs)
        self._create_edges()
        self._add_entry_exit_blocks()
        self._compute_dominators()
        self._identify_loops()
        self._compute_dominance_frontier()
        return self.blocks
    
    def _reset(self):
        """Reset the builder state."""
        self.blocks.clear()
        self.edges.clear()
        self.current_block_id = 0
        self.entry_block_id = None
        self.exit_block_id = None

    def _identify_leaders(self, instrs: List[Dict]):
        """Identify basic block leaders with enhanced criteria."""
        self.blocks.clear()
        current_block = None
        
        for i, instr in enumerate(instrs):
            is_leader = (
                i == 0 or 
                instr.get("op") in ["jmp", "br", "call"] or  # CF instructions
                i > 0 and instrs[i-1].get("op") in ["jmp", "br", "ret", "call"] or  # After CF
                "label" in instr or  # Labels
                (i > 0 and "critical" in instr.get("labels", [])) or  # Critical section
                (i > 0 and self._is_exception_boundary(instr))  # boundaries
            )
            
            if is_leader:
                self.current_block_id += 1
                current_block = BasicBlock(id=self.current_block_id)
                self.blocks[self.current_block_id] = current_block
                
                if self.entry_block_id is None:
                    self.entry_block_id = self.current_block_id
            
            if current_block is not None:
                current_block.instructions.append(instr)

    def _is_exception_boundary(self, instr: Dict) -> bool:
        """Check if instruction marks exception handling boundary."""
        return (
            "try" in instr.get("labels", []) or
            "catch" in instr.get("labels", []) or
            "finally" in instr.get("labels", [])
        )

    def _create_edges(self):
        """Create edges between basic blocks with type classification."""
        for block_id, block in self.blocks.items():
            if not block.instructions:
                continue
            
            last_instr = block.instructions[-1]
            next_block_id = block_id + 1
            
            # Handle different types of cf
            if last_instr.get("op") == "jmp":
                self._handle_jump(block_id, last_instr)
            elif last_instr.get("op") == "br":
                self._handle_branch(block_id, last_instr)
            elif last_instr.get("op") == "ret":
                if self.exit_block_id:
                    self._add_edge(block_id, self.exit_block_id, "fall-through")
            else:
                if next_block_id in self.blocks:
                    self._add_edge(block_id, next_block_id, "fall-through")

    def _handle_jump(self, block_id: int, instr: Dict):
        """Handle unconditional jump instructions."""
        target_label = instr.get("labels", [""])[0]
        target_block = self._find_block_by_label(target_label)
        if target_block:
            edge_type = "back-edge" if target_block.id <= block_id else "branch"
            self._add_edge(block_id, target_block.id, edge_type)

    def _handle_branch(self, block_id: int, instr: Dict):
        """Handle conditional branch instructions."""
        for label in instr.get("labels", []):
            target_block = self._find_block_by_label(label)
            if target_block:
                edge_type = "back-edge" if target_block.id <= block_id else "branch"
                self._add_edge(block_id, target_block.id, edge_type)

    def _add_edge(self, source: int, target: int, edge_type: str):
        """Add edge with type classification."""
        edge = Edge(source, target, edge_type)
        self.edges.append(edge)
        self.blocks[source].successors.add(target)
        self.blocks[target].predecessors.add(source)

    def _add_entry_exit_blocks(self):
        """Add explicit entry and exit blocks."""
        if not self.blocks:
            return

        if not self.entry_block_id:
            self.current_block_id += 1
            self.entry_block_id = self.current_block_id
            self.blocks[self.entry_block_id] = BasicBlock(id=self.entry_block_id)
            first_block = min(self.blocks.keys())
            self._add_edge(self.entry_block_id, first_block, "fall-through")

        self.current_block_id += 1
        self.exit_block_id = self.current_block_id
        self.blocks[self.exit_block_id] = BasicBlock(id=self.exit_block_id)
        
        for block_id, block in self.blocks.items():
            if block.instructions and block.instructions[-1].get("op") == "ret":
                self._add_edge(block_id, self.exit_block_id, "fall-through")

    def _compute_dominators(self):
        """Compute dominators using iterative data-flow algorithm."""
        if not self.entry_block_id:
            return

        #  dominators
        for block in self.blocks.values():
            block.dominators = set(self.blocks.keys())
        
        entry_block = self.blocks[self.entry_block_id]
        entry_block.dominators = {self.entry_block_id}
        
        changed = True
        while changed:
            changed = False
            for block_id, block in self.blocks.items():
                if block_id == self.entry_block_id:
                    continue
                
                #Calculate new dominators from predecessors
                new_doms = None
                for pred_id in block.predecessors:
                    pred_doms = self.blocks[pred_id].dominators
                    if new_doms is None:
                        new_doms = pred_doms.copy()
                    else:
                        new_doms &= pred_doms
                
                if new_doms is not None:
                    new_doms = {block_id} | new_doms
                    if new_doms != block.dominators:
                        block.dominators = new_doms
                        changed = True

        # immediate dominators
        self._compute_immediate_dominators()

    def _compute_immediate_dominators(self):
        """Compute immediate dominators for each block."""
        for block in self.blocks.values():
            dom_candidates = block.dominators - {block.id}
            if dom_candidates:
                idom = None
                for dom_id in dom_candidates:
                    if not any(
                        dom_id in self.blocks[other_dom].dominators 
                        for other_dom in dom_candidates 
                        if other_dom != dom_id
                    ):
                        idom = dom_id
                        break
                block.idom = idom

    def _identify_loops(self):
        """Identify natural loops in the CFG."""
        for edge in self.edges:
            if edge.edge_type == "back-edge":
                header = self.blocks[edge.target]
                header.loop_header = True
                
                loop_nodes = self._find_natural_loop(edge.target, edge.source)
                
                for node_id in loop_nodes:
                    self.blocks[node_id].loop_depth += 1

    def _find_natural_loop(self, header_id: int, tail_id: int) -> Set[int]:
        """Find all nodes in a natural loop."""
        loop_nodes = {header_id, tail_id}
        worklist = {tail_id}
        
        while worklist:
            current = worklist.pop()
            for pred_id in self.blocks[current].predecessors:
                if pred_id not in loop_nodes:
                    loop_nodes.add(pred_id)
                    worklist.add(pred_id)
        
        return loop_nodes

    def _compute_dominance_frontier(self):
        """Compute dominance frontier for each block."""
        for block in self.blocks.values():
            if len(block.predecessors) >= 2:
                for pred_id in block.predecessors:
                    runner = pred_id
                    while runner != block.idom and runner is not None:
                        self.blocks[runner].dom_frontier.add(block.id)
                        runner = self.blocks[runner].idom

    def get_loop_nesting_forest(self) -> Dict[int, Set[int]]:
        """Return the loop nesting forest structure."""
        forest = defaultdict(set)
        for block_id, block in self.blocks.items():
            if block.loop_header:
                current_id = block_id
                max_depth = block.loop_depth
                parent_header = None
                
                while current_id is not None:
                    current = self.blocks[current_id]
                    if current.loop_header and current.loop_depth < max_depth:
                        parent_header = current_id
                        break
                    current_id = current.idom
                
                if parent_header is not None:
                    forest[parent_header].add(block_id)
                else:
                    forest[None].add(block_id)
                    
        return dict(forest)

    def get_dominator_tree(self) -> Dict[int, Set[int]]:
        """Return the dominator tree structure."""
        dom_tree = defaultdict(set)
        for block_id, block in self.blocks.items():
            if block.idom is not None:
                dom_tree[block.idom].add(block_id)
        return dict(dom_tree)

class SSAConverter:
    def __init__(self):
        self.blocks: Dict[int, BasicBlock] = {}
        self.var_versions: DefaultDict[str, int] = defaultdict(int)
        self.var_defs: DefaultDict[str, List[SSADefinition]] = defaultdict(list)
        self.var_uses: DefaultDict[str, Set[int]] = defaultdict(set)
        self.stack: DefaultDict[str, List[int]] = defaultdict(list)
        
    def convert_to_ssa(self, cfg: Dict[int, BasicBlock]) -> Dict[int, BasicBlock]:
        """Convert CFG to SSA form using the standard algorithm."""
        self.blocks = cfg
        self._compute_variable_uses()
        self._insert_phi_functions()
        self._rename_variables()
        return self.blocks
    
    def _compute_variable_uses(self):
        """Find all variables and their uses in the CFG."""
        self.var_uses.clear()
        for block in self.blocks.values():
            self._find_block_variables(block)
    
    def _find_block_variables(self, block: BasicBlock):
        """Find all variable definitions and uses in a block."""
        for instr in block.instructions:
            if "dest" in instr:
                var = instr["dest"]
                self.var_uses[var].add(block.id)
            
            if "args" in instr:
                for arg in instr["args"]:
                    if isinstance(arg, str) and not arg.isdigit():
                        self.var_uses[arg].add(block.id)
    
    def _insert_phi_functions(self):
        """Insert phi functions using dominance frontier."""
        for var in self.var_uses:
            def_blocks = {block_id for block_id in self.var_uses[var]
                         if any("dest" in instr and instr["dest"] == var 
                              for instr in self.blocks[block_id].instructions)}
            
            # Calculate dominance frontier for phi placement
            phi_blocks = set()
            worklist = list(def_blocks)
            while worklist:
                block_id = worklist.pop()
                df = self.blocks[block_id].dominance.dom_frontier
                
                for df_block_id in df:
                    if df_block_id not in phi_blocks:
                        phi_blocks.add(df_block_id)
                        if df_block_id not in def_blocks:
                            worklist.append(df_block_id)
            
            for block_id in phi_blocks:
                self._insert_phi(var, block_id)
    
    def _insert_phi(self, var: str, block_id: int):
        """Insert a phi function for a variable in a specific block."""
        block = self.blocks[block_id]
        
        phi_node = {
            "op": "phi",
            "dest": var,  # will be renamed later
            "args": ["_" for _ in block.predecessors], 
            "labels": list(block.predecessors), 
            "type": self._get_var_type(var)
        }
        
        block.phi_nodes.append(phi_node)
    
    def _get_var_type(self, var: str) -> Optional[Dict]:
        """Get the type of a variable from its definition."""
        for block in self.blocks.values():
            for instr in block.instructions:
                if "dest" in instr and instr["dest"] == var:
                    return instr.get("type")
        return None
    
    def _rename_variables(self):
        """Rename variables to SSA form using depth-first traversal."""
        self.stack.clear()
        entry_block_id = min(self.blocks.keys())
        
        self._rename_block(entry_block_id, set())
    
    def _rename_block(self, block_id: int, visited: Set[int]):
        """Rename variables in a block and recursively process successors."""
        if block_id in visited:
            return
        
        block = self.blocks[block_id]
        visited.add(block_id)
        old_stack_sizes = {var: len(stack) for var, stack in self.stack.items()}
        
        # Rename phi nodes
        for phi in block.phi_nodes:
            var = phi["dest"]
            new_version = self._new_version(var)
            phi["dest"] = f"{var}_{new_version}"
        
        for instr in block.instructions:
            if "args" in instr:
                new_args = []
                for arg in instr["args"]:
                    if isinstance(arg, str) and not arg.isdigit():
                        if self.stack[arg]:
                            current_version = self.stack[arg][-1]
                            new_args.append(f"{arg}_{current_version}")
                        else:
                            new_args.append(arg) 
                    else:
                        new_args.append(arg)
                instr["args"] = new_args
        
            if "dest" in instr:
                var = instr["dest"]
                new_version = self._new_version(var)
                instr["dest"] = f"{var}_{new_version}"
        
        # Process successors and update their phi nodes
        for succ_id in block.successors:
            succ = self.blocks[succ_id]
            pred_index = list(succ.predecessors).index(block_id)
            
            for phi in succ.phi_nodes:
                var = phi["dest"].split('_')[0] 
                if self.stack[var]: 
                    phi["args"][pred_index] = f"{var}_{self.stack[var][-1]}"
        
        # Recursively process successors
        for succ_id in block.successors:
            self._rename_block(succ_id, visited)
        
        # Restore stack sizes
        for var, size in old_stack_sizes.items():
            while len(self.stack[var]) > size:
                self.stack[var].pop()
    
    def _new_version(self, var: str) -> int:
        """Create new version number for a variable and update stack."""
        self.var_versions[var] += 1
        version = self.var_versions[var]
        self.stack[var].append(version)
        return version

    def _is_critical_edge(self, pred_id: int, succ_id: int) -> bool:
        """Check if edge between blocks is critical."""
        pred = self.blocks[pred_id]
        succ = self.blocks[succ_id]
        return len(pred.successors) > 1 and len(succ.predecessors) > 1

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

@dataclass
class ArrayAccess:
    array_name: str
    index_expr: str
    version: int
    block_id: int
    defining_version: Optional[int] = None
    loop_depth: int = 0
    stride: Union[int, Dict[str, int]] = 0

class DataLayoutAnalyzer:
    def __init__(self):
        self.array_info: Dict[str, ArrayInfo] = {}
        self.array_versions: DefaultDict[str, List[ArrayAccess]] = defaultdict(list)
        self.loop_nest_info: Dict[int, LoopInfo] = {}
        self.expr_parser = IndexExpressionParser()
        self.cfg_builder = CFGBuilder()
        self.ssa_converter = SSAConverter()
    
    def analyze_function(self, func: BrilFunction) -> Dict[str, ArrayInfo]:
        print("\nStarting memory access pattern analysis...")
        self._reset_analysis()
        
        cfg = self.cfg_builder.build_cfg(func.instrs)
        print(f"Built CFG with {len(cfg)} basic blocks")
        ssa_cfg = self.ssa_converter.convert_to_ssa(cfg)
        print("Converted to SSA form")
        
        self._collect_array_declarations(func)
        self._analyze_array_accesses_ssa(ssa_cfg)
        patterns = self._determine_access_patterns_ssa()
        self._print_analysis_results(patterns)
        
        return patterns
    
    def _reset_analysis(self):
        self.array_info.clear()
        self.array_versions.clear()
        self.loop_nest_info.clear()
    
    def _analyze_array_accesses_ssa(self, cfg: Dict[int, BasicBlock]):
        """Analyze array accesses using SSA form information."""
        for block in cfg.values():
            loop_depth = self._calculate_loop_depth(block, cfg)
            
            for phi in block.phi_nodes:
                base_name = phi["dest"].split('_')[0]
                if base_name in self.array_info:
                    self._process_phi_node_ssa(phi, block.id, loop_depth)
            
            for instr in block.instructions:
                if instr.get("op") in ["load", "store"]:
                    self._process_array_access_ssa(instr, block.id, loop_depth)
    
    def _process_phi_node_ssa(self, phi: Dict, block_id: int, loop_depth: int):
        """Process phi node with SSA version tracking."""
        base_name = phi["dest"].split('_')[0]
        curr_version = int(phi["dest"].split('_')[1])
        
        phi_access = ArrayAccess(
            array_name=base_name,
            index_expr="phi",
            version=curr_version,
            block_id=block_id,
            loop_depth=loop_depth
        )
        
        for arg in phi["args"]:
            if isinstance(arg, str) and '_' in arg:
                prev_version = int(arg.split('_')[1])
                phi_access.defining_version = prev_version
        
        self.array_versions[base_name].append(phi_access)
    
    def _process_array_access_ssa(self, instr: Dict, block_id: int, loop_depth: int):
        """Process array access with SSA version information."""
        args = instr.get("args", [])
        if len(args) >= 2:
            array_ref = args[0]
            if not isinstance(array_ref, str) or '_' not in array_ref:
                return
                
            array_name, version = array_ref.split('_')
            if array_name not in self.array_info:
                return
                
            index_expr = str(args[1])
            curr_version = int(version)
            
            loop_vars, var_strides = self.expr_parser.parse(index_expr)
            
            access = ArrayAccess(
                array_name=array_name,
                index_expr=index_expr,
                version=curr_version,
                block_id=block_id,
                loop_depth=loop_depth,
                stride=max(var_strides.values(), default=0) if var_strides else 0
            )
            
            prev_accesses = [a for a in self.array_versions[array_name] 
                           if a.version < curr_version]
            if prev_accesses:
                access.defining_version = max(a.version for a in prev_accesses)
            
            self.array_versions[array_name].append(access)
            self.array_info[array_name].total_accesses += 1
    
    def _determine_access_patterns_ssa(self) -> Dict[str, ArrayInfo]:
        """Determine access patterns using SSA version information."""
        print("\nDetermining access patterns using SSA analysis...")
        
        for name, info in self.array_info.items():
            if name not in self.array_versions:
                continue
                
            accesses = self.array_versions[name]
            if not accesses:
                continue
            
            try:
                matrix_dim = int(info.dimensions[0]) if info.dimensions else 1024
            except (IndexError, ValueError):
                matrix_dim = 1024
            
            version_chains = self._build_version_chains(accesses)
            chain_patterns = []
            for chain in version_chains:
                pattern = self._analyze_chain_pattern(chain, matrix_dim)
                chain_patterns.append(pattern)
            
            if chain_patterns:
                pattern_counts = Counter(chain_patterns)
                dominant_pattern = pattern_counts.most_common(1)[0][0]
                info.access_pattern = dominant_pattern
                print(f"Array {name}: {dominant_pattern} (from SSA analysis)")
        
        return self.array_info
    
    def _build_version_chains(self, accesses: List[ArrayAccess]) -> List[List[ArrayAccess]]:
        """Build chains of related SSA versions."""
        chains = []
        used_versions = set()
        
        for access in accesses:
            if access.version in used_versions:
                continue
                
            chain = []
            current = access
            while current:
                chain.append(current)
                used_versions.add(current.version)
                next_accesses = [a for a in accesses 
                               if a.defining_version == current.version
                               and a.version not in used_versions]
                current = next_accesses[0] if next_accesses else None
            
            if chain:
                chains.append(chain)
        
        return chains
    
    def _analyze_chain_pattern(self, chain: List[ArrayAccess], 
                             matrix_dim: int) -> AccessPattern:
        """Analyze access pattern within a version chain."""
        if not chain:
            return AccessPattern.RANDOM
        
        strides = []
        for access in chain:
            if isinstance(access.stride, dict):
                stride_val = max(access.stride.values(), default=0)
            else:
                stride_val = access.stride
            if stride_val > 0:
                strides.append(stride_val)
        
        if not strides:
            return AccessPattern.RANDOM
        
        total = len(strides)
        unit_strides = sum(1 for s in strides if s == 1)
        large_strides = sum(1 for s in strides if s >= matrix_dim)
        
        row_ratio = unit_strides / total if total > 0 else 0
        col_ratio = large_strides / total if total > 0 else 0
        
        if col_ratio >= 0.3:
            return AccessPattern.COLUMN_MAJOR
        elif row_ratio >= 0.3:
            return AccessPattern.ROW_MAJOR
        else:
            return AccessPattern.STRIDED
    
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
            
            dominator_id = min(current_block.dominators - {current_block.id}, default=None)
            current_block = cfg.get(dominator_id) if dominator_id else None
        
        return depth
    
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