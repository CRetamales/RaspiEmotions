"""
- Autor: Carlos Retamales A.
- Fecha: 17-07-2023
- Versión: 1.0.0
- Objetivo: Crear una interacción entre el usuario y computador,
            mediante la captura de emociones y la generación de
            respuestas por parte de GPT-3.5-turbo
"""

"""
__DEFINICIÓN DE LIBRERÍAS__
"""

import os
import openai
import tempfile
import requests
import cv2
import requests
import numpy as np
import ipywidgets as widgets
import time
from IPython.display import Audio, clear_output
from elevenlabs import generate, play, set_api_key, voices, Models
from keras.models import load_model
from playsound import playsound

"""
__DEFINICIÓN DE CONSTANTES__
"""

# Definir el nombre de los archivos de los modelos y las claves de las APIs
model_name_emotions = "fer2013_mini_XCEPTION.102-0.66.hdf5"
model_name_faces = "haarcascade_frontalface_default.xml"
openai_api_key     = "" 
eleven_api_key = "" 

# Cargar el modelo pre-entrenado de reconocimiento de emociones
model = load_model(model_name_emotions)

# Cargar el modelo pre-entrenado para la detección de rostros
face_cascade = cv2.CascadeClassifier(model_name_faces)

# Crear un diccionario para mapear los índices de las emociones a sus etiquetas
# Dado que el modelo fue entrenado las etiquetas en inglés, se utilizarán las etiquetas en inglés
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Configurar las claves de la API
openai.api_key = openai_api_key
set_api_key(eleven_api_key)

# Obtener la lista de voces disponibles
voice_list = voices()
voice_labels = [voice.category + " voice: " + voice.name for voice in voice_list]
voice_id_dropdown = voice_labels[-1]
# Encuentra el índice de la opción seleccionada
selected_voice_index = voice_labels.index(voice_id_dropdown)
selected_voice_id    = voice_list[selected_voice_index].voice_id

# Definir el modelo y el sistema de GPT-3.5-turbo
chatgpt_model = "gpt-3.5-turbo"
chatgpt_system = "You are a helpful therapist on a conversation. Answer should be not too long. Be kind" 

"""
__DEFINICIÓN DE FUNCIONES__
"""

"""
 Funcion que captura la emoción del usuario y devuelve la emoción que más se repitió en 10 segundos
 Entrada: Ninguna
 Salida: String con la emoción que más se repitió
"""

def get_emotion_capture():

    # Inicializar la cámara web
    cap = cv2.VideoCapture(1)

    # Toma de inicio de tiempo
    start_time = time.time()

    # Lista de emociones
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    #Lista donde se guardaran las emociones capturadas en el tiempo
    emotion_list = []

    # Se inicia un ciclo while hasta que pasen 10 segundos
    while True:
        # Leer el cuadro de video actual
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI) correspondiente al rostro
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

            # Normalizar la ROI y cambiar su forma para que coincida con la entrada del modelo
            roi_gray = roi_gray / 255.0
            roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))

            # Realizar la predicción de la emoción
            emotion_prediction = model.predict(roi_gray)

            # Obtener la etiqueta de la emoción con mayor probabilidad
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_label = emotion_labels[emotion_label_arg]

            # Dibujar un rectángulo alrededor del rostro y mostrar la etiqueta de la emoción
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

            # Agregar la etiqueta de la emoción a la lista
            emotion_list.append(emotion_label)


        # Mostrar el cuadro de video con los rostros y las etiquetas de las emociones
        # Al frente de todas las ventanas abiertas
        cv2.imshow('Emotion Detection', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

    # Contar el número de veces que se detectó cada emoción
    emotion_count = [emotion_list.count(emotion) for emotion in emotions]
    # print(emotion_list)
    # Devolvemos la emoción que más se repitió
    return emotion_labels[np.argmax(emotion_count)]


"""
 Funcion para obtener la respuesta por parte de modelo de texto GPT de OpenAI
 Entrada: String con la pregunta o consulta del usuario
 Salida: String con la respuesta del modelo
"""

def get_gpt_response(prompt):
    
    # Obtener la respuesta del modelo
    # Documentación: https://platform.openai.com/docs/api-reference/completions/create
    response = openai.ChatCompletion.create(
        model=chatgpt_model,
        messages=[
            {"role": "system", "content": chatgpt_system},
            {"role": "user", "content": prompt}
        ]
    )
    # Devolver la respuesta del modelo
    return response.choices[0].message.content

"""
Funcion para obtener la respuesta por GPT con la interacción de voz de Eleven
Entrada: String con la pregunta o consulta del usuario
Salida: String con el archivo temporal de audio de la respuesta del modelo
"""

def interact_with_gpt(prompt):
    
    # Obtener la respuesta del modelo
    response_text = get_gpt4_response(prompt)

    # Tamaño de los chunks de audio
    CHUNK_SIZE = 1024

    # URL de la API de Eleven con el ID de la voz seleccionada
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + selected_voice_id

    # Headers de la petición HTTP
    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": eleven_api_key
    }

    # Datos de la petición HTTP
    data = {
      "text": response_text,
      "model_id" : "eleven_multilingual_v1",
      "voice_settings": {
        "stability": 0.6,
        "similarity_boost": 1.0
      }
    }

    # Realizar la petición HTTP
    response = requests.post(url, json=data, headers=headers)

    # Guardar el audio en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
        f.flush()
        temp_filename = f.name

    # Devolver el nombre del archivo temporal
    return temp_filename

"""
Funcion para la interacción continua de modelos
Entrada: Ninguna
Salida: Ninguna
"""

def continuous_interaction():
    # Inicializar el ciclo de interacción continua
    while True:
        # Limpiar la salida
        clear_output(wait=True)
        # Obtener la pregunta del usuario
        prompt = input("Ingresa tu pregunta (o escribe 'exit' para salir): ")
        # Salir si el usuario escribe 'exit'
        if prompt.lower() == 'exit':
            break
        # Obtener la emoción del usuario
        emocion = str(get_emotion_capture())
        # Agregar la emoción a la pregunta
        prompt = "Estoy " + emocion + ". " + prompt
        # Obtener la respuesta del modelo de texto
        audio_file = interact_with_gpt4(prompt)
        # Reproducir el audio
        playsound(audio_file)
        
"""=========================================="""
"""=========== BLOQUE PRINCIPAL ============="""
"""=========================================="""

def main():
    # Inicializar la interacción continua de modelos
    continuous_interaction()
    return 0

# Inicializar el programa
if __name__ == "__main__":
    main()

