import streamlit as st
import torch

from litteragpt.transformer.model import (
    BigramLanguageModel,
    Head,
    MultiHeadAttention,
    FeedForward,
    Block,
)
from litteragpt.transformer.tokenizer import tokenizer
from litteragpt.settings import MODEL_PATH


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model() -> BigramLanguageModel:
    model: BigramLanguageModel = torch.load(
        MODEL_PATH, map_location=torch.device("cpu")
    )
    return model


@st.cache_data(show_spinner="Generando...")
def generar_cadena(input: str, num_sentences: int = 1) -> str:
    # Cargamos el modelo
    model = load_model()

    # Codificamos
    indices = tokenizer.encode(input)
    # Transformmos en tensor
    x = torch.tensor(indices).reshape((1, len(input)))

    for _ in range(num_sentences):
        model.eval()
        with torch.no_grad():
            for i in range(x.shape[0]):
                output = model.generate(idx=x, max_new_tokens=1000)[i].tolist()
            # Para seguir generando tenemos que agregar una dimensión
            x = torch.tensor(output).unsqueeze(0)
    # Al salir del bucle quitamos esa dimension
    x = x.squeeze()
    texto = tokenizer.decode(x.tolist())
    return texto
