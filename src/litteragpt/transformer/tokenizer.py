import json
from pathlib import Path

from litteragpt.settings import RUTA_TOKEN_MAP


class Tokenizer:
    def __init__(self, token_map_path: Path = RUTA_TOKEN_MAP) -> None:
        self.token_map_path = token_map_path
        self.token_map = self._load_token_map()
        self.reverse_token_map: dict[int, str] = self._reverse_dict(self.token_map)

    def _load_token_map(self) -> dict[str, int]:
        """Carga el archivo que mapea los caracteres
        con integers

        Returns
        -------
        dict[str, int]
            _description_
        """
        ruta_token_map = self.token_map_path
        with open(ruta_token_map, "r") as f:
            token_map: dict[str, int] = json.load(f)
        return token_map

    def _reverse_dict(self, dict_to_reverse: dict[str, int]) -> dict[int, str]:
        """Da la vuelta a un diccionario transformando
        keys en values y value sen keys

        Parameters
        ----------
        dict_to_reverse : dict[Any, Any]
            _description_

        Returns
        -------
        dict[Any, Any]
            _description_
        """
        return {int(v): str(k) for k, v in dict_to_reverse.items()}

    @property
    def vocab_size(self) -> int:
        """Devuelve el número de caracteres
        del vocabulario

        Returns
        -------
        int
            _description_
        """
        return len(self.token_map)

    def encode(self, texto: str) -> list[int]:
        """Transforma un texto tipo string en una lista de
        integers

        Parameters
        ----------
        texto : str
            _description_

        Returns
        -------
        list[int]
            _description_
        """
        return [self.token_map[c] for c in texto]

    def decode(self, lista_tokens: list[int]) -> str:
        """Transforma una lista de integers en un texto

        Parameters
        ----------
        lista_tokens : list[int]
            _description_

        Returns
        -------
        str
            _description_
        """
        return "".join([self.reverse_token_map[num] for num in lista_tokens])


tokenizer = Tokenizer()
