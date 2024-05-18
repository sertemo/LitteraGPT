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

import pytest
import torch

from litteragpt.transformer.tokenizer import Tokenizer
from litteragpt.transformer.decoder import BigramLanguageModel
from litteragpt.transformer import config as c


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = Tokenizer()
    return tokenizer

@pytest.fixture
def dummy_input():
    return torch.randint(0, 100, (1, c.block_size, c.n_embd)).float()


@pytest.fixture
def bigram_model():
    return BigramLanguageModel()