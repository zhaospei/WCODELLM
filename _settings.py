import getpass
import os
import sys

__USERNAME = getpass.getuser()
_BASE_DIR = f'/drive2/tuandung/WCODELLM/data'
MODEL_PATH = f'{_BASE_DIR}/weights/'
DATA_FOLDER = os.path.join(_BASE_DIR, 'benchmark')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

