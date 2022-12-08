# LIBRERÍAS:
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
from scipy import signal

# TODO: Pensar en una forma en la que el diccionario de frames no se llene eternamente.
# TODO: Filtrar para que, por ejemplo, cuando se detecte movimiento en dos timestamps MUY cercanos, se tome como uno.

# Carga:
camara = cv2.VideoCapture("SinCobija_Luz.mp4")

fps = camara.get(cv2.CAP_PROP_FPS) # Frames por segundo.

# Inicializa el primer frame a vacío. Servirá para obtener el fondo:
fondo = None

mi_dicc_fr = dict()              # Diccionario {llave: frame_count, valor: frame}

# Listas que contendrán los deltas de cada frame:
suma = list()                    # Delta con el frame inmediatamente anterior.
suma10 = list()                  # Delta con el frame en la décima posición anterior.
suma30 = list()                  # Delta con el frame en la trigésima posición anterior.
suma60 = list()                  # Delta con el frame en la sexagésima posición anterior.

umbral_suma = 0.27  # 27%       # Umbral para detección de picos (donde hay mayor movimiento).

n = 0
frame_count = 0

# Recorre todos los frames:
while True:
    # Obtiene el frame:
    n += 1
    (grabbed, frame) = camara.read()
    if (n % int(fps)) != 0:     # Lee un frame por segundo.
        continue

    frame_count += 1

    # Si se ha llegado al final del video, sale:
    if not grabbed:
        break

    ''' --------------------------------------- PREPARACIÓN FRAME -------------------------------------------'''
    # Convierte a escala de grises el frame actual:
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Suaviza para eliminar ruido:
    gris = cv2.GaussianBlur(gris, (21, 21), 0)

    # Si todavía no se ha obtenido el fondo, se obtendrá aquí. Será el primer frame que se obtenga.
    if fondo is None:
        fondo = gris
        continue

    ''' --------------------------------------- COMPARACIÓN FRAMES -------------------------------------------'''
    mi_dicc_fr[frame_count] = gris   # Va agregando la info al diccionario.

    # Diferencia entre el frame actual y el anterior:
    try:
        resta_frames = cv2.absdiff(gris, mi_dicc_fr[frame_count-1]) # Graficar
        sumita = np.array(resta_frames).sum()
        suma.append(sumita)
    except:
        resta_frames = 0
        sumita = 0

    '''           ********************************************************************************              '''

    # Diferencia entre el frame actual y el décimo anterior:
    try:
        resta_frames10 = cv2.absdiff(gris, mi_dicc_fr[frame_count-10]) # Graficar
        sumita10 = np.array(resta_frames10).sum()
        suma10.append(sumita10)
    except:
        resta_frames10 = 0
        sumita10 = 0

    '''           ********************************************************************************              '''

    # Diferencia entre el frame actual y el trigésimo anterior:
    try:
        resta_frames30 = cv2.absdiff(gris, mi_dicc_fr[frame_count - 30])  # Graficar
        sumita30 = np.array(resta_frames30).sum()
        suma30.append(sumita30)
    except:
        resta_frames30 = 0
        sumita30 = 0

    '''           ********************************************************************************              '''

    # Diferencia entre el frame actual y el sexagésimo anterior:
    try:
        resta_frames60 = cv2.absdiff(gris, mi_dicc_fr[frame_count - 60])  # Graficar
        sumita60 = np.array(resta_frames60).sum()
        suma60.append(sumita60)
    except:
        resta_frames60 = 0
        sumita60 = 0


''' ---------------------------------------------- VISUALIZACIÓN -------------------------------------------------'''

# +++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 1: +++++++++++++++++++++++++++++++++++++++++++++++++

# Normalización señal:
suma = suma - np.min(suma)
suma = suma / np.max(suma)

# Encuentra dónde hubo mayor movimiento:
ind_peaks, _ = signal.find_peaks(suma, height=umbral_suma)

print("Tiempos (mins) en los que se detecta movimiento usando delta con el frame anterior: ")
for k in ind_peaks:
    print(datetime.timedelta(seconds=float(k)))  # Timestamp de los picos.

plt.plot(suma)
plt.plot(ind_peaks, suma[ind_peaks], 'ro')
plt.xlabel('# Frame')
plt.ylabel('Delta frames')
plt.title('Δ (Present frame - Previous frame)')
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 2: +++++++++++++++++++++++++++++++++++++++++++++++++
suma10 = suma10 - np.min(suma10)
suma10 = suma10 / np.max(suma10)

ind_peaks10, _ = signal.find_peaks(suma10, height=umbral_suma)

plt.plot(suma10)
plt.plot(ind_peaks10, suma10[ind_peaks10], 'ro')
plt.xlabel('# Frame')
plt.ylabel('Delta frames')
plt.title('Δ (Present frame - Previous 10 frame)')
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 3: ++++++++++++++++++++++++++++++++++++++++++++++++
suma30 = suma30 - np.min(suma30)
suma30 = suma30 / np.max(suma30)

ind_peaks30, _ = signal.find_peaks(suma30, height=umbral_suma)

plt.plot(suma30)
plt.plot(ind_peaks30, suma30[ind_peaks30], 'ro')
plt.xlabel('# Frame')
plt.ylabel('Delta frames')
plt.title('Δ (Present frame - Previous 30 frame)')
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 4: ++++++++++++++++++++++++++++++++++++++++++++++++
suma60 = suma60 - np.min(suma60)
suma60 = suma60 / np.max(suma60)

ind_peaks60, _ = signal.find_peaks(suma60, height=umbral_suma)

plt.plot(suma60)
plt.plot(ind_peaks60, suma60[ind_peaks60], 'ro')
plt.xlabel('# Frame')
plt.ylabel('Delta frames')
plt.title('Δ (Present frame - Previous 60 frame)')
plt.show()