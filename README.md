# LitteraGPT
### v0.1.0

![Tests](https://github.com/sertemo/LitteraGPT/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/sertemo/LitteraGPT/graph/badge.svg?token=6N7LBN76A2)](https://codecov.io/gh/sertemo/LitteraGPT)
![Dependabot](https://img.shields.io/badge/dependabot-enabled-blue.svg?logo=dependabot)
![GitHub](https://img.shields.io/github/license/sertemo/LitteraGPT)

## Descripción
Aplicación para realizar inferencia con un modelo **Transformer Decoder** basado en la arquitectura transformer según el paper [Attention is all you need](https://arxiv.org/abs/1706.03762).

El Decoder es capaz de predecir el siguiente token a partir de los tokens anteriores de contexto.

La inferencia se realiza con la plataforma **Streamlit**.

## Características del Decoder
### Dataset
El Dataset se ha conformado con 5 obras literarias de la literatura española. Las obras son en prosa y son las siguientes:
- El Quijote
- Lazarillo de Tormes
- Marianela
- Arbol de la ciencia
- La Regenta

Las obras han sido descargadas de la página de [Project Gutenberg](https://www.gutenberg.org/browse/languages/es)

Se han encadenado los textos y quitado los dobles espacios y dobles guiones. Posteriormente se ha dividio el conjunto del texto por frases. Se han creado **chunks** de 20 frases, creando 2001 párrafos para después hacer un **shuffle** de todos los párrafos.

El dataset es un string con **3.568.528** caracteres y un vocabulario de **115** caracteres.

### Tokenizer
**Caracter level** tokenizer

Se mapean todos los 115 caracteres del vocabulario a un integer y se crea una función para codificar y decodificar. Se reserva un **5%** del dataset para validación.

### Modelo
Modelo que sigue la arquitectura del decoder dentro de lo que se conoce como **Transformer**:

![alt text](<assets/img/decoder arquitectura.JPG>)

El decoder utiliza un mecanismo de  **Multi-head** **self attention** y **masked self attention**.

El modelo resultante tiene 19,29 millones de parámetros entrenables.

### Hiperparámetros
| Hiperparámetro | Valor | Descripción                                                  |
|:---------------|:------|:-------------------------------------------------------------|
| Batch          | 64    | Cuantas muestras simultáneas le entran al modelo             |
| Contexto       | 528   | Número de caracteres de cada ejemplo que le entran el modelo |
| Iteraciones    | 10000 | Número de iteraciones realizadas en el entreanmiento         |
| Learning rate  | 3e-4  | ratio  de aprendizaje                                        |
| Embeddings     | 512   | Dimensión del vector de embeddings de cada token             |
| Heads          | 8     | Número de cabezas del decador                                |
| Layers         | 6     | Número de capas del Decoder                                  |
| Dropout        | 0.3   | Dropout utilizado durante el entrenamiento                   |


### Entrenamiento
Se usa Pytorch como framework para realizar el entrenamiento.

Para el entrenamiento se utiliza una **NVIDIA GPU V100** disponible con mi cuenta de *Google Colab**. El tiempo total de entrenamiento es de 2 horas y 7 minutos a 1,64 it/s.

Como optimizador se utiliza **AdamW** y como función de pérdida, loss, se usa **Cross Entropy**.
Se obtiene una **loss** de entrenamiento de **0.9762** y una **loss** de validación de **1,1533**.


## Uso
Escribe una palabra o una frase en la caja de texto y presiona sobre el botón **Generar**. El modelo generará dos frases tomando como contexto la palabra o la frase indicada.

![alt text](<assets/img/uso litteragpt.JPG>)

## Tests
![Pytest](https://img.shields.io/badge/testing-pytest-blue.svg)
![Black](https://img.shields.io/badge/code%20style-black-blue.svg)
![Flake8](https://img.shields.io/badge/linter-flake8-blue.svg)
![MyPy](https://img.shields.io/badge/type%20checker-mypy-blue.svg)

## Tecnologías
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Poetry](https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-black?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Contribuyentes
Este proyecto se ha basado en el video tutorial de Andrew Karpathy en el que monta una estructura similar y la entrena con la obra de Shakespeare. El video está disponible en [youtube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Licencia
Copyright 2024 Sergio Tejedor Moreno

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

