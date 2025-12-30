"""Run evaluation for audio, text, and face models and print classification reports."""
"""Wrapper to run per-modality evaluation scripts sequentially and report results."""
import subprocess, sys
print('Running full evaluation (audio, text, face) by invoking per-modality scripts...')
for s in ['scripts/eval_audio.py','scripts/eval_text.py','scripts/eval_face.py']:
    print(f"Running {s}...")
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        print(f"Script {s} exited with code {r.returncode}")
    else:
        print(f"Completed {s}")
print('Reports saved under models/eval_reports/')

