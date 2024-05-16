import json
from pathlib import Path


class Tokenizer:
    def __init__(self, token_map_filename: str = "token_map.json") -> None:
        self.token_map_filename = token_map_filename
        self.token_map = self._load_token_map()

    def _load_token_map(self) -> dict[str, int]:
        ruta_token_map = Path("src") / "litteragpt" / "ml" / self.token_map_filename
        with open(ruta_token_map, "r") as f:
            token_map = json.load(f)
        return token_map

    def encode(self, texto: str) -> list[int]:
        return [self.token_map[c] for c in texto]

    def decode(self, lista_tokens: list[int]) -> str:
        pass
