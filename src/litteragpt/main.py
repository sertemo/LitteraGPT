import streamlit as st

from litteragpt.transformer.decoder import generar_cadena
from litteragpt.transformer.model import (
    BigramLanguageModel,
    Block,
    MultiHeadAttention,
    Head,
    FeedForward,
)
from litteragpt.settings import N_FRASES_DEFAULT
from litteragpt.styles import CSS_STYLES, Fonts, Sizes
from litteragpt.utils import (
    stream,
    imagen_con_enlace,
    añadir_salto,
    mostrar_enlace,
    texto,
)


def main():
    # Setup de la página
    st.set_page_config(
        page_title="LitteraGPT · Generación de texto basada en la literatura española",
        page_icon="🎭",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    texto(
        "LitteraGPT",
        font_family=Fonts.poppins,
        centrar=True,
        font_size=60,
        formato="b",
        color="#4E4F50",
    )
    texto(
        "GPT que genera un texto con el estilo de varias obras de la literatura española.",
        font_family="Poppins",
        centrar=True,
        color="#746C70",
    )

    mostrar_enlace(
        "Ver detalles del modelo",
        "https://github.com/sertemo/LitteraGPT?tab=readme-ov-file#caracter%C3%ADsticas-del-decoder",
        color="#647C90",
        centrar=True,
    )

    añadir_salto(2)

    input = st.text_input(
        "Escribe una palabra o frase.",
        help="Escribe una palabra o frase.",
        label_visibility="hidden",
    )

    añadir_salto()

    col1, col2, col3 = st.columns(3)
    with col2:
        button_clicked = st.button("Generar Texto", use_container_width=True)

    añadir_salto()

    if button_clicked and input and (input != st.session_state.get("input")):
        cadena = generar_cadena(input, num_sentences=N_FRASES_DEFAULT)
        stream(cadena)
        st.session_state["input"] = input
        st.session_state["response"] = cadena
    else:
        if respuesta := st.session_state.get("response"):
            # Si son el mismo es que se ha recargado la página con los mismos datos
            texto_formateado = f"""
            <div class="output-text" style='font-size: {Sizes.stream_text_size}; \
                color: #4E4F50; font-family: {Fonts.poppins};'>
                {respuesta}
            </div>
            """
            st.markdown(texto_formateado, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
