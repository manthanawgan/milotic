#hydra entry points for the four pipeline stages.

from pathlib import Path

import configs

CONFIG_PATH = str(Path(configs.__file__).resolve().parent)
