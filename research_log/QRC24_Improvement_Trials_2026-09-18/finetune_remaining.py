from pathlib import Path
import subprocess,sys
for seed in (3407,41001):
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('finetune.py')),'--seed',str(seed)],check=True)
