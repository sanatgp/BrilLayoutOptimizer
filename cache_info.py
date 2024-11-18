# cache_info.py

import platform
import subprocess
import re

class CacheInfo:
    def __init__(self):
        self.l1_size = 0
        self.l2_size = 0
        self.l3_size = 0
        self.line_size = 0
        self._detect_cache_sizes()
    
    def _detect_cache_sizes(self):
        system = platform.system()
        if system == "Linux":
            self._detect_linux_cache()
        elif system == "Darwin":
            self._detect_darwin_cache()
        else:
            self._set_default_sizes()
    
    def _detect_linux_cache(self):
        try:
            with open("/sys/devices/system/cpu/cpu0/cache/index0/size", "r") as f:
                self.l1_size = self._parse_size(f.read().strip())
            with open("/sys/devices/system/cpu/cpu0/cache/index2/size", "r") as f:
                self.l2_size = self._parse_size(f.read().strip())
            with open("/sys/devices/system/cpu/cpu0/cache/index3/size", "r") as f:
                self.l3_size = self._parse_size(f.read().strip())
            with open("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r") as f:
                self.line_size = int(f.read().strip())
        except:
            self._set_default_sizes()
    
    def _detect_darwin_cache(self):
        try:
            output = subprocess.check_output(["sysctl", "hw.cachesize"]).decode()
            cache_sizes = re.findall(r"\d+", output)
            if len(cache_sizes) >= 3:
                self.l1_size = int(cache_sizes[0])
                self.l2_size = int(cache_sizes[1])
                self.l3_size = int(cache_sizes[2])
            
            line_size_output = subprocess.check_output(["sysctl", "hw.cachelinesize"]).decode()
            self.line_size = int(re.findall(r"\d+", line_size_output)[0])
        except:
            self._set_default_sizes()
    
    def _set_default_sizes(self):
        self.l1_size = 32 * 1024
        self.l2_size = 256 * 1024
        self.l3_size = 8 * 1024 * 1024
        self.line_size = 64
    
    def _parse_size(self, size_str):
        size = int(size_str[:-1])
        unit = size_str[-1].upper()
        multipliers = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
        return size * multipliers.get(unit, 1)