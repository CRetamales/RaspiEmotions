# RaspiEmotions

## Descripción

Este proyecto es un ejemplo de como utilizar la librería [OpenCV](https://opencv.org/), [TensorFlow](https://www.tensorflow.org/) y [Keras](https://keras.io/) para el reconocimiento facial y de emociones. El proyecto está desarrollado en Python, y se espera una integración para una Raspberry Pi 3B.

El modelo de reconocimiento de emociones fue otorgado por [Oarriaga](https://github.com/oarriaga/face_classification/tree/master/trained_models/emotion_models) para esta primera versión, luego se espera realizar un entrenamiento propio, mediante la metodología de datos de [FER2013](https://paperswithcode.com/dataset/fer2013).


Para la interacción con el usuario se utiliza librerias de OpenIA, como [GPT-3.5-turbo](https://platform.openai.com/docs/guides/gpt) y [Elevenlabs](elevenlabs.io) para la generación de texto y voz, en esta primera versión.

El entorno de desarrollo fue Anaconda, con Python 3.9.X y se utilizó un entorno virtual para la instalación de las librerías.

Finalmente el testeo se realizo mediante imagenes generadas por inteligencia artificial de [Leonardo](https://app.leonardo.ai/).

