# optimization_checker.py

import json
import sys
from typing import Dict, List, Any

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

def compare_files():
    with open('input.json', 'r') as f:
        input_json = json.load(f)
    with open('output.json', 'r') as f:
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