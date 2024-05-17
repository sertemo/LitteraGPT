# Copyright 2024 Sergio Tejedor Moreno

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

from litteragpt.transformer.tokenizer import Tokenizer

def test_tokenizer_load_correct(tokenizer: Tokenizer):
    assert isinstance(tokenizer.token_map, dict)

def test_encode_correct(tokenizer: Tokenizer):
    frase = "Quiero codificar"
    codificado = tokenizer.encode(frase)
    print(codificado)
    assert isinstance(codificado, list), "No es una lista"
    assert all(isinstance(c, int) for c in codificado), "No son todos los índices tipo integer"

def test_vocab_size_correct(tokenizer: Tokenizer):
    assert tokenizer.vocab_size, "No se ha creado el vocab size"
    assert isinstance(tokenizer.vocab_size, int), "El vocab size no se correcto"

def test_decode_correct(tokenizer: Tokenizer):
    tokens = [random.randint(0, tokenizer.vocab_size) for _ in range(10)]
    texto = tokenizer.decode(tokens)
    print(texto)
    assert isinstance(texto, str), "El texto no es un string"

