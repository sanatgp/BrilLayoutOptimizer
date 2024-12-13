import json
import sys
from typing import Dict, List, Any, Tuple

def analyze_patterns(json_obj: Dict[str, Any], prefix: str = "") -> List[str]:
    patterns = []
    if isinstance(json_obj, dict):
        if json_obj.get("op") == "alloc":
            patterns.append(f"{prefix}Found allocation: {json_obj.get('dest')} with type {json_obj.get('type')}")
        
        if json_obj.get("op") == "loop":
            loop_info = f"{prefix}Loop with args: {json_obj.get('args')}"
            patterns.append(loop_info)
            if "body" in json_obj:
                patterns.extend(analyze_patterns(json_obj["body"], prefix + "  "))
        
        for key, value in json_obj.items():
            patterns.extend(analyze_patterns(value, prefix))
            
    elif isinstance(json_obj, list):
        for item in json_obj:
            patterns.extend(analyze_patterns(item, prefix))
            
    return patterns

def detect_loop_unrolling(json_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Detect if loop unrolling has been applied and collect unrolled loop info."""
    unrolled_loops = []
    has_unrolling = False
    
    def analyze_loop_body(body: Dict, loop_var: str) -> bool:
        """Check if a loop body shows signs of unrolling."""
        instrs = body.get("instrs", [])
        var_occurrences = set()
        
        for instr in instrs:
            if "args" in instr:
                for arg in instr["args"]:
                    if isinstance(arg, str) and loop_var in arg:
                        if f"({loop_var} +" in arg:
                            var_occurrences.add(arg)
        
        # If we find multiple offset versions of the loop variable, it's likely unrolled
        return len(var_occurrences) > 1
    
    def recursive_check(obj: Any) -> None:
        nonlocal has_unrolling
        if isinstance(obj, dict):
            if obj.get("op") == "loop":
                args = obj.get("args", [])
                if args and "body" in obj:
                    loop_var = args[0]
                    if analyze_loop_body(obj["body"], loop_var):
                        has_unrolling = True
                        step = args[3] if len(args) > 3 else "1"
                        unrolled_loops.append(
                            f"Loop variable {loop_var} with step {step}"
                        )
            
            for value in obj.values():
                recursive_check(value)
        elif isinstance(obj, list):
            for item in obj:
                recursive_check(item)
    
    recursive_check(json_obj)
    return has_unrolling, unrolled_loops

def compare_files():
    with open('input6.json', 'r') as f:
        input_json = json.load(f)
    with open('output6.json', 'r') as f:
        output_json = json.load(f)
    
    print("Input program structure:")
    input_patterns = analyze_patterns(input_json)
    for pattern in input_patterns:
        print(pattern)
        
    print("\nOutput program structure:")
    output_patterns = analyze_patterns(output_json)
    for pattern in output_patterns:
        print(pattern)
    
    input_loops = str(input_patterns)
    output_loops = str(output_patterns)
    
    print("\nOptimization Analysis:")

    input_unrolled, input_unroll_info = detect_loop_unrolling(input_json)
    output_unrolled, output_unroll_info = detect_loop_unrolling(output_json)
    
    if not input_unrolled and output_unrolled:
        print("✓ Loop unrolling detected")
        print("  Unrolled loops found:")
        for loop in output_unroll_info:
            print(f"  - {loop}")
    else:
        print("✗ No loop unrolling applied")
    
    if "_tile" in output_loops and "_tile" not in input_loops:
        print("✓ Loop tiling detected")
    else:
        print("✗ No loop tiling applied")
    
    if any("alloc" in p for p in output_patterns):
        orig_sizes = [p for p in input_patterns if "alloc" in p]
        new_sizes = [p for p in output_patterns if "alloc" in p]
        if str(orig_sizes) != str(new_sizes):
            print("✓ Array padding detected")
        else:
            print("✗ No array padding applied")
    
    orig_loops = [p for p in input_patterns if "Loop with args" in p]
    new_loops = [p for p in output_patterns if "Loop with args" in p]
    if str(orig_loops) != str(new_loops):
        print("✓ Loop interchange detected")
    else:
        print("✗ No loop interchange applied")

if __name__ == "__main__":
    compare_files()