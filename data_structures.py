# data_structures.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from enum import Enum

class AccessPattern(Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    STRIDED = "strided"
    RANDOM = "random"

@dataclass
class BrilType:
    base: str
    ptr: bool = False
    params: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(type_dict: Union[str, Dict]) -> 'BrilType':
        if isinstance(type_dict, str):
            return BrilType(base=type_dict)
        elif isinstance(type_dict, dict):
            if 'ptr' in type_dict:
                return BrilType(base=type_dict['ptr'], ptr=True)
            return BrilType(base=next(iter(type_dict)))
        raise ValueError(f"Invalid type specification: {type_dict}")

@dataclass
class BrilArgument:
    name: str
    type: BrilType

@dataclass
class BrilFunction:
    name: str
    args: List[BrilArgument]
    type: Optional[BrilType]
    instrs: List[Dict]

@dataclass
class MemoryAccess:
    variable: str
    index_expr: str
    line_number: int
    loop_depth: int
    stride: int = 0
    loop_vars: List[str] = field(default_factory=list)

@dataclass
class ArrayInfo:
    dimensions: List[int]
    access_pattern: AccessPattern
    total_accesses: int
    stride_pattern: List[int]
    element_type: str

@dataclass
class LoopInfo:
    var: str
    start: int
    end: int
    step: int
    body: Dict
    depth: int
    parent: Optional['LoopInfo'] = None
