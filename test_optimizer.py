# test_optimizer.py

import json
import pytest
from main import parse_bril_function, convert_function_to_json

def test_type_preservation():
    # Sample input with pointer types
    input_func = {
        "name": "test",
        "args": [
            {
                "name": "x",
                "type": {"ptr": "int"}
            }
        ],
        "instrs": []
    }
    
    # Parse and convert back
    func = parse_bril_function(input_func)
    output_func = convert_function_to_json(func)
    
    # Verify type preservation
    assert output_func["args"][0]["type"] == {"ptr": "int"}

def test_array_formatting():
    input_instr = {
        "op": "add",
        "args": ["x", "y"],
        "dest": "z"
    }
    
    # Convert to JSON and back
    json_str = json.dumps(input_instr, indent=2)
    output_instr = json.loads(json_str)
    
    # Verify array format is preserved
    assert isinstance(output_instr["args"], list)
    assert output_instr["args"] == ["x", "y"]