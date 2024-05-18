import torch

from litteragpt.transformer.model import (
    Head,
    MultiHeadAttention,
    FeedForward,
    Block,
    BigramLanguageModel
)
from litteragpt.transformer.decoder import load_model, generar_cadena
from litteragpt.transformer import config as c
from litteragpt.transformer.tokenizer import tokenizer


def test_head(dummy_input):
    head_size = c.n_embd // c.n_head
    head = Head(head_size)
    output = head(dummy_input)
    assert output.shape == (1, c.block_size, head_size)


def test_multihead_attention(dummy_input):
    head_size = c.n_embd // c.n_head
    multihead = MultiHeadAttention(c.n_head, head_size)
    output = multihead(dummy_input)
    assert output.shape == (1, c.block_size, c.n_embd)


def test_feedforward(dummy_input):
    ff = FeedForward(c.n_embd)
    output = ff(dummy_input)
    assert output.shape == (1, c.block_size, c.n_embd)


def test_block(dummy_input):
    block = Block(c.n_embd, c.n_head)
    output = block(dummy_input)
    assert output.shape == dummy_input.shape


def test_bigram_language_model(bigram_model):
    idx = torch.randint(0, tokenizer.vocab_size, (1, c.block_size))
    logits, loss = bigram_model(idx)
    assert logits.shape == (1, c.block_size, tokenizer.vocab_size)
    assert (loss is None) or isinstance(loss, torch.Tensor)


def test_generate(bigram_model):
    idx = torch.randint(0, tokenizer.vocab_size, (1, c.block_size))
    generated = bigram_model.generate(idx, max_new_tokens=10)
    assert generated.shape[1] == c.block_size + 10
