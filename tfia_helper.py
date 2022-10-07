import numpy as np
import math
from sklearn import metrics
import seaborn as sns
import tensorflow as tf

ENTRENAMIENTO_BINARIO = "C:\\PATH\\Binario\\"
ENTRENAMIENTO_MULTIPLE = "C:\\PATH\\Multiple\\"
PRUEBA_BINARIO = "C:\\PATH\\Binario\\"
PRUEBA_MULTIPLE = "C:\\PATH\\Multiple\\"

def cargar_entrenamiento_binario():
    return (np.genfromtxt(ENTRENAMIENTO_BINARIO+'Datos.csv', delimiter=','),
        np.genfromtxt(ENTRENAMIENTO_BINARIO+'Etiquetas.csv', delimiter=','))

def cargar_entrenamiento_multiple():
    return (np.genfromtxt(ENTRENAMIENTO_MULTIPLE+'Datos.csv', delimiter=','),
        np.genfromtxt(ENTRENAMIENTO_MULTIPLE+'Etiquetas.csv', delimiter=','))

def cargar_prueba_binario():
    return (np.genfromtxt(PRUEBA_BINARIO+'Datos.csv', delimiter=','),
        np.genfromtxt(PRUEBA_BINARIO+'Etiquetas.csv', delimiter=','))

def cargar_prueba_multiple():
    return (np.genfromtxt(PRUEBA_MULTIPLE+'Datos.csv', delimiter=','),
        np.genfromtxt(PRUEBA_MULTIPLE+'Etiquetas.csv', delimiter=','))

def convertir_dominio_frecuencia(serie, promedio=1, tasa=8192, hamming=True):
    if(hamming):
        serie = serie*np.hamming(tasa)

    serie_fft = abs(np.fft.fft(serie))
    kernel = np.ones(promedio)/promedio
    serie_fft = np.convolve(serie_fft, kernel, mode='valid')
    serie_fft = serie_fft[0:int(tasa/2)]
    
    return serie_fft

def recortar_multiplo(serie, multiplo=8192):
    # Metodo para recortar una secuencia de datos para ser una cantidad multiplo de n
    multiplo_cercano = multiplo*math.floor(len(serie)/multiplo)
    particiones = multiplo_cercano/multiplo
    
    return serie[0:multiplo_cercano], int(particiones)

def ventana_deslizante(serie, tamano=512, tasa=8192):
    # Metodo para recorrer datos con una ventana deslizante
    arreglo = np.empty

    for i in range(int(len(serie)/tamano)):
        inicio = i*tamano
        
        if i == 0:
            arreglo = serie[inicio:inicio+tasa]
        else:
            arreglo = np.concatenate((arreglo, serie[inicio:inicio+tasa]))
        
    return arreglo

def generar_desalineamiento(serie, tasa=8192, tiempo_final=128, frecuencia_fundamental=47):
    # Metodo para generar datos simulados de soltura
    tiempo = np.arange(0, tiempo_final, 1/tasa)
    magnitud_aleatoria = 0.03
    sinusoide = (magnitud_aleatoria 
    * np.sin(2 * np.pi * 2*frecuencia_fundamental * tiempo + 0))
    
    if(serie.ndim > 1):
        for i in range(serie.shape[1]):
            serie[:,i] = serie[:,i] + sinusoide
    else:
        serie = serie + sinusoide

    return serie

def generar_soltura(serie, tasa=8192, tiempo_final=128, frecuencia_fundamental=47):
    # Metodo para generar datos simulados de soltura
    tiempo = np.arange(0, tiempo_final, 1/tasa)
    magnitud_aleatoria = 0.01

    if(serie.ndim > 1):
        for i in range(serie.shape[1]):
            for j in range(3,9):
                sinusoide = (magnitud_aleatoria 
                * np.sin(2 * np.pi * j * frecuencia_fundamental * tiempo + 0))

                serie[:,i] = serie[:,i] + sinusoide
    else:
        for i in range(3,9):
            sinusoide = (magnitud_aleatoria 
            * np.sin(2 * np.pi * i * frecuencia_fundamental * tiempo + 0))
            
            serie = serie + sinusoide

    return serie 

def matriz_confusion_binario(y_prueba, prediccion):
    # Metodo para imprimir una matriz de confusion en clasificacion binaria
    exactitud = metrics.accuracy_score(y_prueba , prediccion)
    precision = metrics.precision_score(y_prueba , prediccion)
    exhaustividad = metrics.recall_score(y_prueba , prediccion)
    valor_f1 = metrics.f1_score(y_prueba , prediccion)
    resultado_matriz = metrics.confusion_matrix(y_prueba , prediccion)

    group_names = ['Verdadero negativo',
                    'Falso positivo',
                    'Falso negativo',
                    'Verdadero positivo']
    group_counts = ["{0:0.0f}".format(valor) for valor in
                    resultado_matriz.flatten()]
    group_percentages = ["{0:.2%}".format(valor) for valor in
                        resultado_matriz.flatten()/np.sum(resultado_matriz)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(resultado_matriz, annot=labels, fmt='', cmap='Blues')

    print('Clasificador binario')
    print('Exactitud: '+str("{:.2f}".format(exactitud)))
    print('Precision: '+str("{:.2f}".format(precision)))
    print('Exhaustividad: '+str("{:.2f}".format(exhaustividad)))
    print('Valor F1: '+str("{:.2f}".format(valor_f1)))

def matriz_confusion_multiple(y_prueba, prediccion):
    # Metodo para imprimir una matriz de confusion en clasificacion multiple
    resultado_matriz = metrics.confusion_matrix(y_prueba,np.argmax(prediccion,axis=-1))

    group_counts = ["{0:0.0f}".format(value) for value in
                    resultado_matriz.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        resultado_matriz.flatten()/np.sum(resultado_matriz)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
            zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(4,4)

    sns.heatmap(resultado_matriz,
                annot=labels,
                fmt='',
                cmap='Blues').set(xlabel='Predicciones', ylabel='Categorias')

def guardar_modelo(model, output_path, nombre):
    # Metodo para guardar un modelo
    model_json = model.to_json()
    
    with open(output_path+nombre+'.json','w') as json_file:
        json_file.write(model_json)

    model.save_weights(output_path+nombre+'.h5')

def cargar_modelo(input_path, nombre):
    # Metodo para cargar un modelo
    json_file = open(input_path+nombre+'.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(input_path+nombre+'.h5')

    return model

def sampling(distribucion):
    # Metodo para forzar la salida de un encoder a una distribucion normal
    encoder_media, encoder_varianza = distribucion
    epsilon = tf.keras.backend.random_normal(
        shape=(tf.keras.backend.shape(encoder_media)[0], 2),
        mean=0,
        stddev=0.1)

    return encoder_media + tf.keras.backend.exp(encoder_varianza) * epsilon

