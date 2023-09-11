# -*- coding: utf-8 -*-
"""
En este archivo se trabajarÁ en una CLIMATOLOGÍA CONJUNTA entre 
datos de precipitación de Ezeiza-SMN e Índice SIS

Período: 01-10-1979 al 30-04-2016

Posteriormente se actualizará la base de datos hasta la actualidad

@author: Usuario
"""
import os
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import MultipleLocator 
register_matplotlib_converters()
sns.set() # Setting seaborn as default style even if use only matplotlib

#estamos probando git

#%%
directorio = 'data_set'
archivo= 'datos_combinados.csv'
fname = os.path.join(directorio,archivo)
datos_combinados = pd.read_csv(fname)

datos_combinados['Fecha'] = pd.to_datetime(datos_combinados['Fecha'])
datos_combinados.index = datos_combinados['Fecha']
#Hasta aqui en el df tengo a Fecha como columna y como indice.
#Por el momento lo dejaré así quizas me sirva para algunas operaciones
#me voy a quedar solo con dias de lluvia para poder hacer las estadisticas
dcomb_solo_lluvia= datos_combinados[datos_combinados['prcp']!=0] #de 13362 datos solo me quedan 3356
#%%
'''quiero crear una función que tome como argumentos las condiciones que deseo aplicar en el análisis.  
Se utilizan argumentos opcionales (`sis_condition` y `periodo_condition`). Si no se proporciona un valor para
uno de los argumentos, esa condición no se aplicará en el filtrado de datos. 
Esto me permitirá reutilizar el mismo procedimiento con diferentes conjuntos de condiciones sin tener que repetir el código,
incluyendo casos donde solo una de las condiciones es relevante.'
La cree en un archivo de funciones y la voy a importar aqui para utilizarla.'''
from funciones_para_pp_y_sis import climatologia_conjunta
# Llamada a la función con diferentes condiciones
#EJEMPLO
result = climatologia_conjunta(dcomb_solo_lluvia, periodo_condition='f', prcp_condition=15)
'''Si quiero contestar a la pregunta: 
                  
                ¿CUÁL ES LA CLIMATOLOGÍA DE LA PRECIPITACIÓN PARA EVENTOS POSITIVOS DEL SIS ESTIVAL?
'''


result1 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='positivos', periodo_condition='c')
result1.to_excel('plot/clim_conj_1.xlsx', index=True)
result2 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='positivos', periodo_condition='f')
result2.to_excel('plot/clim_conj_2.xlsx', index=True)
#junto para graficar las 2 tablas de resultado
combined_result= pd.concat([result1, result2], axis=1)
combined_result=combined_result.drop(['Count', 'Min'], axis=0)

#plotting
indice=combined_result.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Precipitación en eventos POSITIVOS del SIS\nPeríodo 1979-2016', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2, combined_result.iloc[:, 0], 
       width=ancho_barras, color='darkorange', label="SIS-ESTIVAL")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_result.iloc[:, 1], width=ancho_barras,
       color='burlywood', label="SIS-INVERNAL")

# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")
ax.legend(loc='upper left', fontsize=9)


ax.set_ylabel('Precipitación diaria (mm)', fontsize=11)

'''
#pone etiqueta a cada barra
# Coordenadas donde se ubicarán las anotaciones en la parte superior
xytext_top = (0, 10)
# Coordenadas donde se ubicarán las anotaciones en la parte inferior
xytext_bottom = (0, -11)
color_top= 'red'
color_bottom= 'blue'
for p in ax.patches:
    if p.get_height()>0:
        va = 'top'  # Anotación en la parte superior
        xytext = xytext_top
        color= color_top
    else:
        va = 'bottom'  # Anotación en la parte inferior
        xytext = xytext_bottom
        color= color_bottom
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va=va,  xytext=xytext, textcoords='offset points', fontsize=9, color= color, fontweight='normal')
'''
plt.savefig('plot/climat_c_pp_sispos.png', dpi=200)
plt.show()
#%%
'''Si quiero contestar a la pregunta: 
                  
                ¿CUÁL ES LA CLIMATOLOGÍA DE LA PRECIPITACIÓN PARA EVENTOS DEL SIS ESTIVAL?
                
                POSITIVOS VS NEGATIVOS
'''


result1 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='positivos', periodo_condition='c')
#result1.to_excel('plot/clim_conj_1.xlsx', index=True)
result2 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='negativos', periodo_condition='c')
#result2.to_excel('plot/clim_conj_2.xlsx', index=True)
#junto para graficar las 2 tablas de resultado
combined_result= pd.concat([result1, result2], axis=1)
combined_result=combined_result.drop(['Count', 'Min'], axis=0)

#plotting
indice=combined_result.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Precipitación durante la época ESTIVAL del patrón SIS\nPeríodo 1979-2016', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2, combined_result.iloc[:, 0], 
       width=ancho_barras, color='darkorange', label="SIS-POSITIVOS")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_result.iloc[:, 1], width=ancho_barras,
       color='burlywood', label="SIS-NEGATIVOS")

# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")
ax.legend(loc='upper left', fontsize=9)


ax.set_ylabel('Precipitación diaria (mm)', fontsize=11)

'''
#pone etiqueta a cada barra
# Coordenadas donde se ubicarán las anotaciones en la parte superior
xytext_top = (0, 10)
# Coordenadas donde se ubicarán las anotaciones en la parte inferior
xytext_bottom = (0, -11)
color_top= 'red'
color_bottom= 'blue'
for p in ax.patches:
    if p.get_height()>0:
        va = 'top'  # Anotación en la parte superior
        xytext = xytext_top
        color= color_top
    else:
        va = 'bottom'  # Anotación en la parte inferior
        xytext = xytext_bottom
        color= color_bottom
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va=va,  xytext=xytext, textcoords='offset points', fontsize=9, color= color, fontweight='normal')
'''
plt.savefig('plot/climat_c_pp_sisESTIVAL.png', dpi=200)
plt.show()

#%%
'''Si quiero contestar a la pregunta: 
                  
                ¿CUÁL ES LA CLIMATOLOGÍA DE LA PRECIPITACIÓN PARA EVENTOS DEL SIS INVERNAL?
                
                POSITIVOS VS NEGATIVOS
'''


result1 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='positivos', periodo_condition='f')
#result1.to_excel('plot/clim_conj_1.xlsx', index=True)
result2 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='negativos', periodo_condition='f')
#result2.to_excel('plot/clim_conj_2.xlsx', index=True)
#junto para graficar las 2 tablas de resultado
combined_result= pd.concat([result1, result2], axis=1)
combined_result=combined_result.drop(['Count', 'Min'], axis=0)

#plotting
indice=combined_result.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Precipitación durante la época INVERNAL del patrón SIS\nPeríodo 1979-2016', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2, combined_result.iloc[:, 0], 
       width=ancho_barras, color='cadetblue', label="SIS-POSITIVOS")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_result.iloc[:, 1], width=ancho_barras,
       color='skyblue', label="SIS-NEGATIVOS")

# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")
ax.legend(loc='upper left', fontsize=9)


ax.set_ylabel('Precipitación diaria (mm)', fontsize=11)

'''
#pone etiqueta a cada barra
# Coordenadas donde se ubicarán las anotaciones en la parte superior
xytext_top = (0, 10)
# Coordenadas donde se ubicarán las anotaciones en la parte inferior
xytext_bottom = (0, -11)
color_top= 'red'
color_bottom= 'blue'
for p in ax.patches:
    if p.get_height()>0:
        va = 'top'  # Anotación en la parte superior
        xytext = xytext_top
        color= color_top
    else:
        va = 'bottom'  # Anotación en la parte inferior
        xytext = xytext_bottom
        color= color_bottom
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va=va,  xytext=xytext, textcoords='offset points', fontsize=9, color= color, fontweight='normal')
'''
plt.savefig('plot/climat_c_pp_sisINVERNAL.png', dpi=200)
plt.show()
#%%
'''Si quiero contestar a la pregunta: 
                  
                ¿CUÁL ES LA CLIMATOLOGÍA DE LA PRECIPITACIÓN PARA EVENTOS DEL SIS ?
                
                POSITIVOS VS NEGATIVOS
'''


result1 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='positivos')
#result1.to_excel('plot/clim_conj_1.xlsx', index=True)
result2 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='negativos')
#result2.to_excel('plot/clim_conj_2.xlsx', index=True)
#junto para graficar las 2 tablas de resultado
combined_result= pd.concat([result1, result2], axis=1)
combined_result=combined_result.drop(['Count', 'Min'], axis=0)

#plotting
indice=combined_result.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Precipitación en base a patrón SIS positivos y negativos\nPeríodo 1979-2016', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2, combined_result.iloc[:, 0], 
       width=ancho_barras, color='red', label="SIS-POSITIVOS")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_result.iloc[:, 1], width=ancho_barras,
       color='violet', label="SIS-NEGATIVOS")

# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")
ax.legend(loc='upper left', fontsize=9)


ax.set_ylabel('Precipitación diaria (mm)', fontsize=11)

'''
#pone etiqueta a cada barra
# Coordenadas donde se ubicarán las anotaciones en la parte superior
xytext_top = (0, 10)
# Coordenadas donde se ubicarán las anotaciones en la parte inferior
xytext_bottom = (0, -11)
color_top= 'red'
color_bottom= 'blue'
for p in ax.patches:
    if p.get_height()>0:
        va = 'top'  # Anotación en la parte superior
        xytext = xytext_top
        color= color_top
    else:
        va = 'bottom'  # Anotación en la parte inferior
        xytext = xytext_bottom
        color= color_bottom
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va=va,  xytext=xytext, textcoords='offset points', fontsize=9, color= color, fontweight='normal')
'''
plt.savefig('plot/climat_c_pp_sis_posandneg.png', dpi=200)
plt.show()
#%%
'''Si quiero contestar a la pregunta: 
                  
                ¿CUÁL ES LA CLIMATOLOGÍA DE LA PRECIPITACIÓN PARA EVENTOS NEGATIVOS DEL SIS?
'''
result3 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='negativos', periodo_condition='c')
result3.to_excel('plot/clim_conj_3.xlsx', index=True)
result4 = climatologia_conjunta(dcomb_solo_lluvia, sis_condition='negativos', periodo_condition='f')
result4.to_excel('plot/clim_conj_4.xlsx', index=True)
#junto para graficar las 2 tablas de resultado
combined_result_neg= pd.concat([result3, result4], axis=1)
combined_result_neg=combined_result_neg.drop(['Count', 'Min'], axis=0)

#plotting
indice=combined_result_neg.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Precipitación en eventos NEGATIVOS del SIS\nPeríodo 1979-2016', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2, combined_result_neg.iloc[:, 0], 
       width=ancho_barras, color='cadetblue', label="SIS-ESTIVAL")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_result_neg.iloc[:, 1], width=ancho_barras,
       color='skyblue', label="SIS-INVERNAL")

# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")
ax.legend(loc='upper left', fontsize=9)


ax.set_ylabel('Precipitación diaria (mm)', fontsize=11)

'''
#pone etiqueta a cada barra
# Coordenadas donde se ubicarán las anotaciones en la parte superior
xytext_top = (0, 10)
# Coordenadas donde se ubicarán las anotaciones en la parte inferior
xytext_bottom = (0, -11)
color_top= 'red'
color_bottom= 'blue'
for p in ax.patches:
    if p.get_height()>0:
        va = 'top'  # Anotación en la parte superior
        xytext = xytext_top
        color= color_top
    else:
        va = 'bottom'  # Anotación en la parte inferior
        xytext = xytext_bottom
        color= color_bottom
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va=va,  xytext=xytext, textcoords='offset points', fontsize=9, color= color, fontweight='normal')
'''
plt.savefig('plot/climat_c_pp_sisneg.png', dpi=200)
plt.show()
#%%
archivo2= 'intensidadpp_mensual.csv'
fname = os.path.join(directorio,archivo2)
intensidadpp_mensual = pd.read_csv(fname)

intensidadpp_mensual['Fecha'] = pd.to_datetime(intensidadpp_mensual['Fecha'])
intensidadpp_mensual.index = intensidadpp_mensual['Fecha']
#%%


''' CALCULO DE FRECUENCIAS'''


#%%

# Calcular la cantidad de días con sis positivo, ACA USO INCLUSO LOS DATOS CON CERO DE PP
sis_positive_days = datos_combinados[datos_combinados['Valor'] > 0]['Fecha'].nunique()
print("De un total de 13.362 datos: Cantidad de días con sis positivo:", sis_positive_days)

positive_sis_days = datos_combinados[datos_combinados['Valor'] > 0]
# Convertir la columna 'Fecha' a tipo datetime
datos_combinados['Fecha'] = pd.to_datetime(datos_combinados['Fecha'])
# Calcular la frecuencia mensual
monthly_frequency = positive_sis_days.groupby(pd.Grouper(key='Fecha', freq='M'))['Fecha'].count()
monthly_frequency.plot(figsize=(25,8), color='orange')


# Título del gráfico
plt.title('Frecuencia Mensual de Días con SIS Positivo', fontsize=20)

# Leyenda
plt.legend(['SIS POSITIVOS'], loc='upper right')
# Tamaño de fuente de los ejes x e y
plt.xlabel('Fecha', fontsize=18)
plt.ylabel('Frecuencia', fontsize=18)

# Tamaño de fuente de los ticks de los ejes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# Mostrar el gráfico
plt.show()
# Imprimir los resultados
print(monthly_frequency)
#%%

# Calcular la cantidad de días con sis positivo, ACA USO INCLUSO LOS DATOS CON CERO DE PP
sis_negative_days = datos_combinados[datos_combinados['Valor'] < 0]['Fecha'].nunique()
print("De un total de 13.362 datos: Cantidad de días con sis negativo:", sis_negative_days)

negative_sis_days = datos_combinados[datos_combinados['Valor'] < 0]
# Convertir la columna 'Fecha' a tipo datetime
datos_combinados['Fecha'] = pd.to_datetime(datos_combinados['Fecha'])
# Calcular la frecuencia mensual
monthly_frequency2 = negative_sis_days.groupby(pd.Grouper(key='Fecha', freq='M'))['Fecha'].count()
monthly_frequency2.plot(figsize=(25,8), color='green')

# Título del gráfico
plt.title('Frecuencia Mensual de Días con SIS Negativo', fontsize=20)

# Leyenda
plt.legend(['SIS NEGATIVOS'], loc='upper right')
# Tamaño de fuente de los ejes x e y
plt.xlabel('Fecha', fontsize=18)
plt.ylabel('Frecuencia', fontsize=18)

# Tamaño de fuente de los ticks de los ejes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# Mostrar el gráfico
plt.show()
# Imprimir los resultados
print(monthly_frequency2)
#%%

# Calcular la cantidad de días con prcp mayor a 20
prcp_above_20_days = df[df['prcp'] > 20]['fecha'].nunique()
print("Cantidad de días con prcp mayor a 20:", prcp_above_20_days)
