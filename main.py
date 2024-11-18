# main.py

import json
import sys
import argparse
import logging
from typing import Dict, Any, Union
from data_structures import BrilFunction, BrilType, BrilArgument
from analysis import DataLayoutAnalyzer
from optimization import LayoutOptimizer

def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )

def preserve_type_info(type_info: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    """Preserve the original type information structure."""
    if isinstance(type_info, str):
        return type_info
    elif isinstance(type_info, dict):
        if "ptr" in type_info:
            return {"ptr": type_info["ptr"]}
    return type_info


def parse_bril_function(func_json: dict) -> BrilFunction:
    """Parse a Bril function from JSON, preserving type information."""
    args = []
    for arg in func_json.get("args", []):
        bril_type = BrilType.from_dict(arg["type"])
        args.append(BrilArgument(arg["name"], bril_type))
    
    return BrilFunction(
        name=func_json.get("name", "unknown"),
        args=args,
        type=BrilType.from_dict(func_json.get("type", "void")) if "type" in func_json else None,
        instrs=func_json.get("instrs", [])
    )

def convert_function_to_json(func: BrilFunction) -> dict:
    """Convert a BrilFunction back to JSON format, preserving original structure."""
    return {
        "name": func.name,
        "args": [
            {
                "name": arg.name,
                "type": {"ptr": arg.type.base} if arg.type.ptr else arg.type.base
            }
            for arg in func.args
        ],
        "instrs": func.instrs
    }

class BrilEncoder(json.JSONEncoder):
    """Custom JSON encoder to maintain consistent array formatting."""
    def default(self, obj):
        if isinstance(obj, (BrilFunction, BrilType, BrilArgument)):
            return obj.__dict__
        return super().default(obj)

def main():
    parser = argparse.ArgumentParser(description="Bril Data Layout Optimizer")
    parser.add_argument("input", nargs="?", type=argparse.FileType("r"), 
                       default=sys.stdin, help="Input Bril JSON file")
    parser.add_argument("-o", "--output", type=argparse.FileType("w"), 
                       default=sys.stdout, help="Output file")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        logging.debug("Reading input program...")
        program = json.load(args.input)
        
        logging.debug("Initializing optimizer components...")
        analyzer = DataLayoutAnalyzer()
        optimizer = LayoutOptimizer(analyzer)
        
        logging.debug("Processing functions...")
        optimized_program = {"functions": []}
        for func_json in program.get("functions", []):
            logging.debug(f"Optimizing function: {func_json.get('name', 'unknown')}")
            
            func = parse_bril_function(func_json)
            
            optimized_func = optimizer.optimize(func)
            
            optimized_func_json = convert_function_to_json(optimized_func)
            optimized_program["functions"].append(optimized_func_json)
        
        logging.debug("Writing optimized program...")
        json.dump(
            optimized_program,
            args.output,
            indent=2,
            cls=BrilEncoder,
            separators=(',', ': ')
        )
        
        logging.debug("Optimization complete.")
        
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()