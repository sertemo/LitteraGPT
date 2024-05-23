from pathlib import Path


# Rutas
TOKEN_MAP_FILENAME = "token_map.json"
RUTA_TOKEN_MAP = Path("src") / "litteragpt" / "transformer" / TOKEN_MAP_FILENAME
MODEL_FILENAME = "LitteraGPT_64_528_10000_esp.pt"
MODEL_PATH = Path("model") / MODEL_FILENAME

N_FRASES_DEFAULT = 1
