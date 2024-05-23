"""Funciones auxiliares para la aplicación de streamlit"""

import time

import streamlit as st

from litteragpt.styles import DEFAULT_COLOR, Fonts, Sizes


def stream(cadena: str) -> None:
    """Streamea la cadena"""
    stream_container = st.empty()
    output = ""
    for letter in cadena:
        output += letter
        texto_formateado = f"""
        <div class="output-text" style='font-size: {Sizes.stream_text_size};\
            color: #4E4F50; font-family: {Fonts.poppins};'>
            {output}
        </div>
        """
        stream_container.markdown(texto_formateado, unsafe_allow_html=True)
        time.sleep(0.05)


def html_button_generate() -> st.markdown:
    return st.markdown(
        '<div class="button-container"><button class="custom-button" \
                onclick="streamlit.sendCustomMessage(\'generate-text\', {})">Generar Texto</button></div>',
        unsafe_allow_html=True,
    )


def texto(
    texto: str,
    /,
    *,
    font_size: int = 30,
    color: str = DEFAULT_COLOR,
    font_family: str = "Helvetica",
    formato: str = "",
    centrar: bool = False,
) -> None:
    """Función para personalizar el texto con HTML"""
    if formato:
        texto = f"<{formato}>{texto}</{formato}>"
    if centrar:
        texto = f"""
                    <div style='text-align: center'>
                        {texto}
                    </div>
                    """
    texto_formateado = f"""<div style='font-size: {font_size}px; color: {color}; \
        font-family: {font_family}'>{texto}</div>"""
    st.markdown(texto_formateado, unsafe_allow_html=True)


def mostrar_enlace(
    label: str,
    url: str,
    *,
    color: str = DEFAULT_COLOR,
    font_size: str = "16px",
    font_family: str = "Poppins",
    centrar: bool = False,
) -> None:
    """Muestra un enlace personalizado.

    Args:
    label (str): El texto que se mostrará como el enlace.
    url (str): La URL a la que apunta el enlace.
    color (str): Color del texto del enlace.
    font_size (str): Tamaño del texto del enlace.
    centrar (bool): Centra el texto
    """
    html = f'<a href="{url}" target="_blank" style="color: {color}; font-size: {font_size}; \
        font-family: {font_family}; text-decoration: none;">{label}</a>'
    if centrar:
        html = f"""
                    <div style='text-align: center'>
                        {html}
                    </div>
                    """
    st.markdown(html, unsafe_allow_html=True)


def imagen_con_enlace(
    url_imagen: str,
    url_enlace: str,
    alt_text: str = "Imagen",
    max_width: int = 60,
    centrar: bool = False,
    radio_borde: int = 15,
) -> None:
    """Muestra una imagen que es también un hipervínculo en Streamlit con bordes redondeados.

    Args:
    url_imagen (str): URL de la imagen a mostrar.
    url_enlace (str): URL a la que el enlace de la imagen debe dirigir.
    alt_text (str): Texto alternativo para la imagen.
    max_width (int): Ancho máximo de la imagen como porcentaje.
    centrar (bool): Si es verdadero, centra la imagen.
    radio_borde (int): Radio del borde redondeado en píxeles.
    """
    html = f'<a href="{url_enlace}" target="_blank"><img src="{url_imagen}" \
        alt="{alt_text}" style="max-width:{max_width}%; height:auto; border-radius:{radio_borde}px;"></a>'
    if centrar:
        html = f"""
                    <div style='text-align: center'>
                        {html}
                    </div>
                    """
    st.markdown(html, unsafe_allow_html=True)


def añadir_salto(num_saltos: int = 1) -> None:
    """Añade <br> en forma de HTML para agregar espacio"""
    saltos = f"{num_saltos * '<br>'}"
    st.markdown(saltos, unsafe_allow_html=True)
