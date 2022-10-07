import tfia_helper as tfia
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import metrics

# Usar el CPU en vez del GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(tf.config.list_physical_devices('GPU'))

# Estilo
plt.style.use('ggplot')

# Constantes
tasa = 8192 #Hz
top_caracteristicas = 1000
f = 47 #Frecuencia fundamental
output_path = 'C:\\OUTPUT_PATH\\'
path_prueba = 'C:\\PATH\\prueba_completo.csv'

# MODELO CLASIFICACION SVM
normal_data_path = 'C:\\PATH\\normal_x_completo.csv'
cav_data_path = 'C:\\PATH\\cavitacion_x_completo.csv'
normal_data = np.genfromtxt(normal_data_path, delimiter=';')
cav_data = np.genfromtxt(cav_data_path, delimiter=';')

# Determinacion de la dependencia de las variables (frecuencias) 
# y las respuestas (categorias)
normal = tfia.ventana_deslizante(normal_data[:,1], tamano=2048)
cavitacion = tfia.ventana_deslizante(cav_data[:,1], tamano=2048)
x_normal, normal_particiones = tfia.recortar_multiplo(normal)
x_cavitacion, cavitacion_particiones = tfia.recortar_multiplo(cavitacion)
x_total = np.concatenate((x_normal,x_cavitacion))
particiones = normal_particiones+cavitacion_particiones
y_normal = np.zeros(normal_particiones)
y_cavitacion = np.ones(cavitacion_particiones)
y_entrenamiento = np.concatenate((y_normal,y_cavitacion))
x_tiempo_total = np.concatenate((x_normal,x_cavitacion))
x_tiempo_total = x_total.reshape(particiones,tasa)

for i in range(len(x_tiempo_total)):   
    if i == 0:
        x_frecuencia = tfia.convertir_dominio_frecuencia(x_tiempo_total[i,:],promedio=5)
    else:
        temp = tfia.convertir_dominio_frecuencia(x_tiempo_total[i,:],promedio=5)
        x_frecuencia = np.concatenate((x_frecuencia,temp))
        
x_frecuencia = x_frecuencia.reshape(particiones,int(tasa/2))

# Prueba-F ANOVA
idx = np.random.permutation(particiones)
seleccion_caracteristicas = SelectKBest(score_func=f_classif, k=top_caracteristicas)
seleccion_caracteristicas.fit(x_frecuencia,y_entrenamiento)

x_frecuencias_dependientes = seleccion_caracteristicas.fit_transform(x_frecuencia,
                                                                    y_entrenamiento)

# Normalizacion
scaler = MinMaxScaler()
scaler.fit(x_frecuencias_dependientes)
x_frecuencias_dependientes = scaler.transform(x_frecuencias_dependientes)

# Shuffle
x_entrenamiento,x_prueba,y_entrenamiento,y_prueba= train_test_split(
    x_frecuencias_dependientes,
    y_entrenamiento,
    test_size=0.20,
    random_state=1,
    stratify=y_entrenamiento)

contador_entrenamiento = np.bincount(y_entrenamiento.astype(int))
contador_prueba = np.bincount(y_prueba.astype(int))

clf = SVC(kernel='linear')
clf.fit(x_entrenamiento,y_entrenamiento)
y_prediccion = clf.predict(x_prueba)

# Plots
print("Exactitud: ",metrics.accuracy_score(y_prueba, y_prediccion))
print("Precision: ",metrics.precision_score(y_prueba, y_prediccion))
print("Exhaustividad: ",metrics.recall_score(y_prueba, y_prediccion))
print(metrics.classification_report(y_prueba, y_prediccion))

# MODELO CLASIFICACION LDA
lda = LinearDiscriminantAnalysis()
lda_componentes_principales = lda.fit_transform(x_entrenamiento, y_entrenamiento)
y_dataframe = pd.DataFrame(y_entrenamiento, columns = ['categoria'])
principal_df = pd.DataFrame(data=lda_componentes_principales, columns=['LDA 1'])
final_df = pd.concat([principal_df, y_dataframe['categoria']], axis=1)
normal = final_df.where(final_df['categoria'] == 0)["LDA 1"].dropna()
cavitacion = final_df.where(final_df['categoria'] == 1)["LDA 1"].dropna()

# Plots
plt.hist([normal.to_list(),cavitacion.to_list()],
    np.linspace(-30, 30, 100),
    label=['Normal', 'Cavitacion'])
plt.legend(loc='upper left')
plt.xlabel('Categorias')
plt.ylabel('Muestras')

# MODELO CLASIFICACION CONV1D
x_train,y_train = tfia.cargar_entrenamiento_binario()
x_train = x_train.reshape(409,4096,3)

model = Sequential()
model.add(Conv1D(64,32, activation='relu', padding='same', strides=2,
    input_shape=x_train.shape[1:], name='Entrada_Conv1d'))
model.add(Conv1D(32,16, activation='relu', padding='valid', strides=2,
    name='Salida_Conv1d'))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(3, activation="relu"))
model.add(Dense(1, activation="sigmoid" , name='Salida_Clasificacion'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
history = model.fit(x_train, y_train, validation_split=0.25,
    epochs=100, callbacks=[callback], verbose=0)
print(model.summary())

# Guardar el modelo
tfia.guardar_modelo(model, output_path, 'clasificacion_CONV1D')

# Plots
plt.plot(history.history['accuracy'], label='Datos de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Datos de validacion cruzada')
plt.xlabel("Epoch")
plt.ylabel("Exactitud")
plt.title("Evolucion de la exactitud durante el entrenamiento")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Datos de entrenamiento')
plt.plot(history.history['val_loss'], label='Datos de validacion cruzada')
plt.xlabel("Epoch")
plt.ylabel("Funcion de perdida")
plt.title("Evolucion de la funcion de perdida durante el entrenamiento")
plt.legend()
plt.show()

# Analisis del error
x_prueba, y_prueba = tfia.cargar_prueba_binario()
x_prueba = x_prueba.reshape(103,4096,3)
prediccion = model.predict(x_prueba)

# Valor AUC
auc_score=metrics.roc_auc_score(y_prueba , prediccion)

# Plot
fpr, tpr, thresholds = metrics.roc_curve(y_prueba , prediccion)
plt.plot(fpr,tpr, label='Modelo - AUC: '+str("{0:.2%}".format(auc_score)))
plt.plot([0,1], [0,1],linestyle='dashed', color='grey')
plt.title("Curva ROC")
plt.xlabel('Tasa de falsos positivos') 
plt.ylabel('Tasa de verdaderos positivos') 
plt.legend()
prediccion = np.round(prediccion)
tfia.matriz_confusion_binario(y_prueba,prediccion)

# MODELO CLASIFICACION LSTM
model = Sequential()
model.add(Conv1D(64,32, activation='relu',padding='valid', strides=2,
    input_shape=(x_train.shape[1],x_train.shape[2]), name='Entrada_Conv1d'))
model.add(LSTM(128, return_sequences = True, name='Entrada_LSTM'))
model.add(LSTM(64, return_sequences = True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid", name='Salida_Clasificacion'))
model.compile(loss='binary_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
history = model.fit(x_train, y_train, validation_split=0.25, 
    epochs=100, callbacks=[callback], verbose=0)
print(model.summary())

# Guardar el modelo
tfia.guardar_modelo(model, output_path, 'clasificacion_LSTM')

# Plots
plt.plot(history.history['accuracy'], label='Datos de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Datos de validacion cruzada')
plt.xlabel("Epoch")
plt.ylabel("Exactitud")
plt.title("Evolucion de la exactitud durante el entrenamiento")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Datos de entrenamiento')
plt.plot(history.history['val_loss'], label='Datos de validacion cruzada')
plt.xlabel("Epoch")
plt.ylabel("Funcion de perdida")
plt.title("Evolucion de la funcion de perdida durante el entrenamiento")
plt.legend()
plt.show()

# Analisis del error
x_prueba, y_prueba = tfia.cargar_prueba_binario()
x_prueba = x_prueba.reshape(103,4096,3)
prediccion = model.predict(x_prueba)

# Valor AUC
auc_score=metrics.roc_auc_score(y_prueba , prediccion)

# Plot
fpr, tpr, thresholds = metrics.roc_curve(y_prueba , prediccion)
plt.plot(fpr,tpr, label='Modelo - AUC: '+str("{0:.2%}".format(auc_score)))
plt.plot([0,1], [0,1],linestyle='dashed', color='grey')
plt.title("Curva ROC")
plt.xlabel('Tasa de falsos positivos') 
plt.ylabel('Tasa de verdaderos positivos') 
plt.legend()
prediccion = np.round(prediccion)
tfia.matriz_confusion_binario(y_prueba,prediccion)

# MODELO DE DETECCION DE ANOMALIAS CON CONV1D
normal_data_path = 'C:\\PATH\\normal_completo.csv'
normal_data = np.genfromtxt(normal_data_path, delimiter=',')
data = normal_data.reshape(256, 4096, 3) # particiones, pasos, caracteristicas

# Enconder
model = Sequential()
model.add(Conv1D(64,32,activation='relu',padding='same',
    strides=2,input_shape=data.shape[1:],name='Entrada_Encoder'))
model.add(Conv1D(32,16, activation='relu', padding='same', strides=2, name='Salida_Encoder'))
# Decoder
model.add(Conv1DTranspose(32,16,
    activation='relu',padding='same', strides=2, name='Entrada_Decoder'))
model.add(Conv1DTranspose(64,32, activation='relu',padding='same', strides=2))
model.add(Conv1DTranspose(data.shape[2],32,padding='same', name='Salida_Decoder'))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
history = model.fit(data, data, validation_split=0.2, 
    epochs=100, callbacks=[callback], verbose=0)
print(model.summary())

# Guardar el modelo
tfia.guardar_modelo(model, output_path, 'detector_anomalias_CONV1D')

# Plots
plt.plot(history.history['loss'], label='Datos de entrenamiento')
plt.plot(history.history['val_loss'], label='Datos de validacion cruzada')
plt.xlabel("Epochs")
plt.ylabel("Funcion de perdida")
plt.title("Evolucion de la funcion de perdida durante el entrenamiento")
plt.legend()

prediccion = model.predict(data)
error = np.mean(np.abs(prediccion-data) , axis=1)
error_distribucion = error.reshape(error.shape[0]*error.shape[1])

plt.plot(error_distribucion)
plt.ylabel('Error absoluto medio (MAE)')
plt.xlabel('Tiempo (s)')

plt.hist(error_distribucion,bins=30)
plt.xlabel('Error absoluto medio (MAE)')
plt.ylabel('Muestras')

data_prueba = np.genfromtxt(path_prueba, delimiter=',')
data_prueba = data_prueba.reshape(256, 4096, 3)
detector_prueba = model.predict(data_prueba)
error = np.mean(np.abs(detector_prueba-data) , axis=1)
error_distribucion_prueba = error.reshape(error.shape[0]*error.shape[1])

plt.plot(error_distribucion_prueba)
plt.plot([0,800],[0.01,0.01], color='black', label='Umbral a 1%')
plt.xlabel("Muestras en tiempo")
plt.ylabel("Error absoluto medio (MAE)")
plt.title("Detector de anomalias")
plt.legend()

# MODELO DE DETECCION DE ANOMALIAS CON LSTM
data = normal_data.reshape(4096, 3, 256) # pasos, caracteristicas, particiones

# Enconder
model = Sequential()
model.add(LSTM(1024, return_sequences = True,
    input_shape=(data.shape[1],data.shape[2]), name='Entrada_Encoder'))
model.add(LSTM(512, return_sequences = False))
model.add(RepeatVector(data.shape[1], name='Salida_Encoder'))
# Decoder
model.add(LSTM(512, return_sequences = True, name='Entrada_Decoder'))
model.add(LSTM(1024, return_sequences = True))
model.add(TimeDistributed(Dense(data.shape[2]), name='Salida_Decoder'))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
history = model.fit(data, data, validation_split=0.2,
    epochs=100, callbacks=[callback], verbose=0)
print(model.summary())

# Guardar el modelo
tfia.guardar_modelo(model, output_path, 'detector_anomalias_LSTM')

# Plots
plt.plot(history.history['loss'], label='Datos de entrenamiento')
plt.plot(history.history['val_loss'], label='Datos de validacion cruzada')
plt.xlabel("Epochs")
plt.ylabel("Funcion de perdida")
plt.title("Evolucion de la funcion de perdida durante el entrenamiento")
plt.legend()

prediccion = model.predict(data)
error = np.mean(np.abs(prediccion-data) , axis=1)
error_distribucion = error.reshape(error.shape[0]*error.shape[1])

plt.plot(error_distribucion)
plt.ylabel('Error absoluto medio (MAE)')
plt.xlabel('Tiempo (s)')

plt.hist(error_distribucion,bins=30)
plt.xlabel('Error absoluto medio (MAE)')
plt.ylabel('Muestras')

data_prueba = np.genfromtxt(path_prueba, delimiter=',')
data_prueba = data_prueba.reshape(4096, 3, 256)
detector_prueba = model.predict(data_prueba)
error = np.mean(np.abs(detector_prueba-data) , axis=1)
error_distribucion_prueba = error.reshape(error.shape[0]*error.shape[1])

plt.plot(error_distribucion_prueba)
plt.plot([0,1100000],[0.1,0.1], color='black', label='Umbral a 10%')
plt.xlabel("Muestras en tiempo")
plt.ylabel("Error absoluto medio (MAE)")
plt.title("Detector de anomalias")
plt.legend()