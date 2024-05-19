import pytest
from streamlit.testing.v1 import AppTest



def test_main():
    """Test the main function of the Streamlit app"""
    # Inicializar la app desde el archivo main.py
    at = AppTest.from_file("src/litteragpt/main.py").run()
    
    # Simular la interacción con el text_input
    at.text_input[0].input("Era una mañana de").run()
    
    # Verificar que se genera la cadena y se muestra correctamente
    cadena_generada = at.session_state["response"]
    assert isinstance(cadena_generada, str)
    assert len(cadena_generada) > len("Era una mañana de")

    # Verificar el estado de la sesión
    assert at.session_state["input"] == "Era una mañana de"

def test_generar_cadena():
    """Test the generar_cadena function"""
    from litteragpt.transformer.decoder import generar_cadena
    
    input_str = "Esto es una prueba"
    output = generar_cadena(input_str, num_sentences=1)
    assert isinstance(output, str)
    assert len(output) > len(input_str)

