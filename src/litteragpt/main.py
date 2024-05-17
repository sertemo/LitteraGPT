import time

import streamlit as st

from litteragpt.transformer.decoder import generar_cadena
from litteragpt.transformer.decoder import BigramLanguageModel, Block, MultiHeadAttention, Head, FeedForward

def stream(cadena: str) -> None:
    """Streamea la cadena"""
    stream_container = st.empty()
    with stream_container:
        output = ""
        for letter in cadena:
            output += letter
            st.subheader(output)
            time.sleep(0.05)


def main():
    # Setup de la página
    st.set_page_config(
        page_title="LitteraGPT",
        page_icon="🎭",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("LitteraGPT")
    st.header(
        "GPT que infiere un texto con el estilo de varias obras de la literatura española."
    )

    num_frases: int = st.slider("Número de frases a inferir", min_value=1, max_value=5)
    input = st.text_input(
        "Escribe una palabra o frase y pulsa **Intro** para que el modelo continúe."
    )

    if input:
        cadena = generar_cadena(input, num_sentences=num_frases)
        stream(cadena)


if __name__ == "__main__":
    main()
