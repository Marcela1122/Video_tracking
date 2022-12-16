# LIBRERÍAS:
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
from skimage.metrics import structural_similarity

# TODO: Cambiar el umbral para detectar cambio de posición.

# Carga:
camara = cv2.VideoCapture("SinCobija_Luz.mp4")

fps = camara.get(cv2.CAP_PROP_FPS)             # Frames por segundo.
ancho = camara.get(cv2.CAP_PROP_FRAME_WIDTH)   # Ancho de fotograma.
largo = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Largo de fotograma.

# Inicializa el primer frame a vacío. Servirá para obtener el fondo:
fondo = None

# Contiene la información de si se detectó cambio en la posición del paciente:
cambio = False

mi_dicc_fr = deque()              # Contiene los últimos 60 frames.
frames = deque()                  # Contiene los frames específicos en los que se ha detectado movimiento.

# Listas que contendrán los deltas de cada frame:
suma = list()                    # Delta con el frame inmediatamente anterior.
suma10 = list()                  # Delta con el frame en la décima posición anterior.
suma30 = list()                  # Delta con el frame en la trigésima posición anterior.
suma60 = list()                  # Delta con el frame en la sexagésima posición anterior.


umbral_suma = 0.30  # 30%       # Umbral para detección de picos (donde hay mayor movimiento).

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
    mi_dicc_fr.append(gris)  # Va agregando el frame.

    # Diferencia entre el frame actual y el anterior:
    try:
        resta_frames = cv2.absdiff(gris, mi_dicc_fr[-2]) # Graficar
        sumita = np.array(resta_frames).sum()
        suma.append(sumita)
    except:
        resta_frames = 0
        sumita = 0

    '''           ********************************************************************************              '''

    # Diferencia entre el frame actual y el décimo anterior:
    try:
        resta_frames10 = cv2.absdiff(gris, mi_dicc_fr[-10]) # Graficar
        sumita10 = np.array(resta_frames10).sum()
        suma10.append(sumita10)
    except:
        resta_frames10 = 0
        sumita10 = 0

    '''           ********************************************************************************              '''

    # Diferencia entre el frame actual y el trigésimo anterior:
    try:
        resta_frames30 = cv2.absdiff(gris, mi_dicc_fr[-30])  # Graficar
        sumita30 = np.array(resta_frames30).sum()
        suma30.append(sumita30)
    except:
        resta_frames30 = 0
        sumita30 = 0

    '''           ********************************************************************************              '''

    # Diferencia entre el frame actual y el sexagésimo anterior:
    try:
        resta_frames60 = cv2.absdiff(gris, mi_dicc_fr[-60])  # Graficar
        sumita60 = np.array(resta_frames60).sum()
        suma60.append(sumita60)
    except:
        resta_frames60 = 0
        sumita60 = 0

    # Garantiza que el tamaño de la variable que contiene los frames sea siempre limitado.
    if len(mi_dicc_fr) > 60:
        mi_dicc_fr.popleft()

''' ---------------------------------------------- VISUALIZACIÓN -------------------------------------------------'''

# +++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 1: +++++++++++++++++++++++++++++++++++++++++++++++++

# Normalización señal:
suma = suma - np.min(suma)
suma = suma / np.max(suma)

#  Non-weighted moving average:
def moving_average(a, n) :
    mov = np.cumsum(a, dtype=float)
    mov[n:] = mov[n:] - mov[:-n]
    return mov[n - 1:] / n


suma = moving_average(suma, 4)

# Encuentra dónde hubo mayor movimiento:
ind_peaks, _ = signal.find_peaks(suma, height=umbral_suma)

# Encuentra la intersección entre la gráfica y un umbral dado. Sirve para saber dónde empieza y termina el movimiento.
idx = np.argwhere(np.diff(np.sign(suma - umbral_suma))).flatten()

# Comparación del frame de inicio de movimiento con el del final. Si son iguales, no hubo cambio de posición.
for k in range(len(idx)):
    camara.set(1, idx[k])
    ret, frame = camara.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (21, 21), 0)

    frames.append(frame)

    if (k % 2) != 0:
        (score, diff) = structural_similarity(frame[k], frame[k-1], full=True)
        if score < 0.9993:
            cambio = True

        print("Inicio: ", datetime.timedelta(seconds=float(idx[k-1])-2), "  | Final: ",
              datetime.timedelta(seconds=float(idx[k])+2), "  | Duración (s): ", idx[k] - idx[k-1],
              "  | Cambio de posición: ", cambio)
    frames.popleft()

# Gráfica:
plt.plot(suma)
plt.plot(ind_peaks, suma[ind_peaks], 'ro')
plt.plot([0, len(suma)], [umbral_suma, umbral_suma], color='green', linestyle='dashed')
plt.xlabel('# Frame')
plt.ylabel('Delta frames')
plt.title('Δ (Present frame - Previous frame)')
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 2: +++++++++++++++++++++++++++++++++++++++++++++++++
#suma10 = suma10 - np.min(suma10)
#suma10 = suma10 / np.max(suma10)

#ind_peaks10, _ = signal.find_peaks(suma10, height=umbral_suma)

#plt.plot(suma10)
#plt.plot(ind_peaks10, suma10[ind_peaks10], 'ro')
#plt.xlabel('# Frame')
#plt.ylabel('Delta frames')
#plt.title('Δ (Present frame - Previous 10 frame)')
#plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 3: ++++++++++++++++++++++++++++++++++++++++++++++++
#suma30 = suma30 - np.min(suma30)
#suma30 = suma30 / np.max(suma30)

#ind_peaks30, _ = signal.find_peaks(suma30, height=umbral_suma)

#plt.plot(suma30)
#plt.plot(ind_peaks30, suma30[ind_peaks30], 'ro')
#plt.xlabel('# Frame')
#plt.ylabel('Delta frames')
#plt.title('Δ (Present frame - Previous 30 frame)')
#plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++ GRÁFICA 4: ++++++++++++++++++++++++++++++++++++++++++++++++
#suma60 = suma60 - np.min(suma60)
#suma60 = suma60 / np.max(suma60)

#ind_peaks60, _ = signal.find_peaks(suma60, height=umbral_suma)

#plt.plot(suma60)
#plt.plot(ind_peaks60, suma60[ind_peaks60], 'ro')
#plt.xlabel('# Frame')
#plt.ylabel('Delta frames')
#plt.title('Δ (Present frame - Previous 60 frame)')
#plt.show()