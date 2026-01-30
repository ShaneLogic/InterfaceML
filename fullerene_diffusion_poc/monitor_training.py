#!/usr/bin/env python3
"""Monitor training progress."""
import time
from pathlib import Path

log_file = Path('training_log.txt')
print("监控训练进度... (Ctrl+C退出)")

last_pos = 0
while True:
    if log_file.exists():
        with open(log_file, 'r') as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            last_pos = f.tell()
            
            for line in new_lines:
                # 只显示关键信息
                if any(kw in line for kw in ['Epoch', 'Loss', 'bond', 'sphere', 'noise', 'Error', 'Traceback', '完成']):
                    print(line.strip())
    
    time.sleep(5)
