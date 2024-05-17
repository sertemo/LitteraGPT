import streamlit as st
import torch

from litteragpt.transformer.decoder import BigramLanguageModel
from litteragpt.transformer.tokenizer import tokenizer
from litteragpt.settings import MODEL_PATH


@st.cache_resource()
def load_model() -> BigramLanguageModel:
    model = BigramLanguageModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model


def main():
    # Setup de la página
    st.set_page_config(
        page_title="LitteraGPT",
        page_icon="🎭",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("LitteraGPT")
    st.subheader(
        "GPT que infiere un texto con el estilo de varias obras de la literatura española."
    )

    st.slider("Número de frases a inferir", min_value=1, max_value=5)
    input = st.text_input(
        "Escribe una palabra o frase y pulsa **Intro** para que el modelo continúe."
    )

    # TODO Hacer que streamee los caracteres.
    if input:
        tokens = tokenizer.encode(input)
        st.write(tokens)


if __name__ == "__main__":
    main()
