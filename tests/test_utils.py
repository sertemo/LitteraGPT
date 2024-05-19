
from streamlit.testing.v1 import AppTest
from litteragpt.utils import stream

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
        stream(output)
    
    assert at.text[0] == cadena
