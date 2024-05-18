import time

import streamlit as st

from litteragpt.transformer.decoder import generar_cadena
from litteragpt.transformer.model import (
    BigramLanguageModel,
    Block,
    MultiHeadAttention,
    Head,
    FeedForward,
)


def stream(cadena: str) -> None:
    """Streamea la cadena"""
    stream_container = st.empty()
    with stream_container:
        output = ""
        for letter in cadena:
            output += letter
            st.write(output)
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
        "GPT que genera un texto con el estilo de varias obras de la literatura española."
    )

    num_frases: int = st.slider("Número de frases a generar", min_value=1, max_value=3)
    input = st.text_input(
        "Escribe una palabra o frase y pulsa **Intro** para que el modelo continúe."
    )

    if input:
        # Comprobamos que el input o el numero de frases no sean el mismo
        if input != st.session_state.get("input") or num_frases != st.session_state.get(
            "num_frases"
        ):
            cadena = generar_cadena(input, num_sentences=num_frases)
            stream(cadena)
            st.session_state["input"] = input
            st.session_state["response"] = cadena
            st.session_state["num_frases"] = num_frases
        else:
            # Si son el mismo es que se ha recargado la página con los mismos datos
            st.write(st.session_state.get("response"))

    st.session_state


if __name__ == "__main__":
    main()
