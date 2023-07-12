import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo pre-entrenado de reconocimiento de emociones
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')

# Cargar el modelo pre-entrenado para la detección de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Crear un diccionario para mapear los índices de las emociones a sus etiquetas
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Inicializar la cámara web
cap = cv2.VideoCapture(1)

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

    # Mostrar el cuadro de video con los rostros y las etiquetas de las emociones
    cv2.imshow('Emotion Detection', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
