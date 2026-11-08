"""Load MediaPipe through an ASCII package path on Windows.

MediaPipe 0.10.x native resource loading can fail for Chinese install paths.
Copy its installed package (including licenses and bundled model files) to a
versioned local app cache, then import it normally. No model downloads required.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import sys


def load_mediapipe():
    spec = importlib.util.find_spec('mediapipe')
    if spec is None or spec.origin is None:
        raise RuntimeError('MediaPipe missing. Use the project environment.')
    source = Path(spec.origin).parent
    if os.name == 'nt' and not str(source).isascii():
        cache = Path(os.environ.get('LOCALAPPDATA',str(Path.home()))) / 'ARLens' / 'mediapipe-0.10.21'
        if not str(cache).isascii():
            raise RuntimeError('MediaPipe needs an ASCII installation path; install the environment in an ASCII directory.')
        package = cache / 'mediapipe'
        marker = cache / 'copy-complete.txt'
        if not marker.exists():
            cache.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,package,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
            marker.write_text(str(source),encoding='utf-8')
        sys.path.insert(0,str(cache))
    import mediapipe
    return mediapipe
