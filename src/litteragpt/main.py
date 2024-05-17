import streamlit as st

from litteragpt.transformer.tokenizer import tokenizer


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
