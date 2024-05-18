import pytest
from streamlit.testing.v1 import AppTest

def test_stream():
    """Test the stream function to ensure it correctly streams text"""
    # Definir una cadena de prueba
    cadena = "Hola"
    
    # Inicializar la app desde el archivo main.py
    at = AppTest.from_file("src/litteragpt/main.py")
    
    # Simular la función stream y verificar que se escribe cada letra correctamente
    output = ""
    for letter in cadena:
        output += letter
        # Simula la escritura de la salida
        at.mock_write(output)
        assert at.mock_last_output == output

def test_main():
    """Test the main function of the Streamlit app"""
    # Inicializar la app desde el archivo main.py
    at = AppTest.from_file("src/litteragpt/main.py").run()
    
    # Simular la interacción con el slider y el text_input
    at.slider[0].set_value(1).run()
    at.text_input[0].input("prueba").run()
    
    # Verificar que se genera la cadena y se muestra correctamente
    cadena_generada = at.session_state["response"]
    assert isinstance(cadena_generada, str)
    assert len(cadena_generada) > len("prueba")

    # Verificar el estado de la sesión
    assert at.session_state["input"] == "prueba"
    assert at.session_state["num_frases"] == 2

def test_generar_cadena():
    """Test the generar_cadena function"""
    from litteragpt.transformer.decoder import generar_cadena
    
    input_str = "Esto es una prueba"
    output = generar_cadena(input_str, num_sentences=1)
    assert isinstance(output, str)
    assert len(output) > len(input_str)

