
# -*- coding: utf-8 -*-
"""
Primer analisis exploratorio de datos de precipitación Ezeiza-SMN e índice SIS hasta abril de 2016
"""
#%% IMPORTO LIBRERIAS
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
#%% ABRO DATOS DE PRECIPITACION VIEJOS
directorio = 'data_set'
archivo= 'ppSMN.csv'
fname = os.path.join(directorio,archivo)
df = pd.read_csv(fname)
df = df.rename(columns={'timeend':'Fecha'})
#primero convierte la col de fecha en datetime y luego la elije como indice y le queda tmb como colimna para usarla con groupby
df['Fecha'] = pd.to_datetime(df['Fecha'])
#voy a poner todas las horas como 12 utc
df['Fecha']= df['Fecha'].dt.normalize()+datetime.timedelta(hours=12)
df.index = df['Fecha']

#ELIMINO LA COLUMNA FECHA Y HAGO DATETIME EL INDICE FECHA COMO NADIA 
df.drop(columns=["Fecha"], axis=1, inplace=True)
df.index = pd.to_datetime(df.index)
#%% DATOS HASTA LA ACTUALIDAD
'''
voy a agregar el df_e con datos actuales al script y 
'''
archivo= 'registros_E.csv'
fname = os.path.join(directorio,archivo)
df_e = pd.read_csv(fname, sep='\t')
df_e.drop(columns=["fecha","omm_id", "tmax", "tmin", "tmed", "td", "pres_est", "pres_nm", "hr", "helio", "nub", "vmax_d", "vmax_f", "vmed", "estado", "num_observaciones"], axis=1, inplace=True)
dates_d = pd.date_range(start='20200102', end='20230321', freq='D')

df_e['Fecha'] =pd.to_datetime(dates_d,format ='%Y/%m/%d')
df_e['Fecha'] = df_e['Fecha'].dt.normalize()+datetime.timedelta(hours=12)
df_e.index = df_e['Fecha']
'''
hasta aca cambie el rango de fecha para que coincida con 
las 12 utc del dia siguiente tal cual estan los datos de pp originales, nadia
previo a comparar con el otro df original, voy a analizar un poco estos datos

''' 
df_e['Fecha']=pd.to_datetime(df_e['Fecha'])
df_e.drop(columns=["Fecha"], axis=1, inplace=True)

df = df.asfreq("D") #me aseguro que esten todas las fechas del rango de datos EN AMBOS DF
df_e=df_e.asfreq("D")
#CANTIDAD DE NAN´S
print(df.isnull().sum())
nan_rows = df[df.isnull().any(1)] # -----> 21 datos NaN

#CANTIDAD DE NAN'S PARA DF_E
print(df_e.isnull().sum())
nan_rows_e = df_e[df_e.isnull().any(1)] 
#%% comparo fechas coincidentes
'''
comparar ambos df en las fechas coincidentes con merge
'''

df = df.rename(columns={'valor':'prcp'})
'''
USO MERGE PARA COMPARAR df con df_e
'''
merge_pp=pd.merge(df, df_e, how='inner', left_index=True, right_index=True)
fin_merge=datetime.datetime(2020, 5, 6)
inicio_merge= datetime.datetime(2020, 1, 2)
duracion_m= fin_merge - inicio_merge
print(duracion_m) # coincide con el rango de fechas que se solapan, porque es 125 y le debo sumar 1
''' cambio valores en prcp LOS CAMBIE EN EL ARCHIVO MANUALMENTE :/
recien ahora puedo eliminar las ultimas filas de 'valor' y adicionarle el df_e '''

df=df.loc['1961-1-2':'2020-1-1'] #--->este es el rango que me quedo del original
df.columns=['prcp']
df=df.append(df_e) # --->AHORA MI DF ES UNO SOLO CON TODOS LOS DATOS, SERIE COMPLETA
#22724 datos
date= pd.date_range(start='19610102', end='20230321', freq='D') #coincide con
#la longitud del df asi que esta completo en fecha
#AHORA LA PUEDO TRABAJAR
#%% OBTENGO UN SOLO DF PARA PRECIPITACIÓN
'''
copio y extiendo datos a actualidad y me quedo con un solo df que va a ser la suma de los
2 y ahi ya puedo aplicar prom movil y hacer climatologia

Cantidad de datos y datos faltantes SERIE COMPLETA
'''
#datos faltantes: True seria que el elemnto es nulo
df.info() 
df.dtypes
df.isna().sum() # cantidad de elementos nulos: 20 NaN's
nan_rows = df[df.isnull().any(1)] 
print(nan_rows)
'''
                     prcp
Fecha                    
1962-02-25 12:00:00   NaN
1962-03-01 12:00:00   NaN
1964-09-11 12:00:00   NaN
1964-09-12 12:00:00   NaN
1964-10-13 12:00:00   NaN
1964-10-14 12:00:00   NaN
2002-01-01 12:00:00   NaN
2006-01-01 12:00:00   NaN
2014-01-01 12:00:00   NaN
2016-01-01 12:00:00   NaN
2016-10-02 12:00:00   NaN
2016-11-29 12:00:00   NaN
2017-01-22 12:00:00   NaN
2018-03-23 12:00:00   NaN
2018-04-29 12:00:00   NaN
2018-08-23 12:00:00   NaN
2018-08-26 12:00:00   NaN
2019-07-23 12:00:00   NaN
2019-08-18 12:00:00   NaN
2020-01-30 12:00:00   NaN

'''
#%% prueba de plotting SERIE COMPLETA

'''plotting SERIE COMPLETA PERO SIN DIAS DE "NO LLUVIA"'''

f_solo_lluvia = df[df['prcp']!=0]
print(f_solo_lluvia.info())
#DatetimeIndex: 5581 entries, 1961-01-02 12:00:00 to 2023-03-20 12:00:00
#prcp    5561 non-null   float64
#dtypes: float64(1)   20 NaN'S

'''aca quiero antes completar lo que seria el eje x para q al graficar me 
quede completo aunque con lugares blancos los dias de no lluvia '''
f_solo_lluvia=f_solo_lluvia.asfreq('D')
#lta añadir una fila mas con valor nan al f_df_lluvia
#result = pd.Series(data={pd.to_datetime('20230321',format ='%Y/%m/%d)})
#df = df.append(result, ignore_index=False)
f_solo_lluvia.index = pd.to_datetime(f_solo_lluvia.index)
print(f_solo_lluvia.info())

f_solo_lluvia.describe()

#pruebo con solo unos datos, tipo lupa
x=f_solo_lluvia.loc['1983-01-02':'1983-12-01']
# area+add a stronger line on top (edge)
# Change the style of plot
plt.figure(figsize=(50,12))
plt.bar(x.index,
       x['prcp'], color="skyblue")
plt.fill_between(x.index,
       x['prcp'], color="skyblue") #, alpha=0.3)
plt.plot(x.index,
       x['prcp'], color="skyblue")

# Show the graph
plt.show()
##%%
#ahora pruebo con toda la serie
plt.figure(figsize=(50,12))
#plt.bar(f_solo_lluvia.index,
       #f_solo_lluvia['prcp'], color="skyblue")
plt.fill_between(f_solo_lluvia.index,
       f_solo_lluvia['prcp'], color="skyblue") #, alpha=0.6)
#plt.plot(f_solo_lluvia.index,
      # f_solo_lluvia['prcp'], color="skyblue")
#%% Plotting SERIE COMPLETA
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import MultipleLocator
register_matplotlib_converters()


# Create figure and plot space
fig, ax = plt.subplots(figsize=(50, 12), facecolor='lavender')
fig.suptitle('Ezeiza SMN - Serie de precipitación diaria\nEnero 1961 - Marzo 2023', fontsize= 28)
# Add x-axis and y-axis
ax.fill_between(f_solo_lluvia.index,
       f_solo_lluvia['prcp'],
       color='blue')

ax.xaxis.set_major_locator(MultipleLocator(731))
ax.xaxis.set_minor_locator(MultipleLocator(365))

# Set title and labels for axes
ax.set(xlim=["1961-01", "2023-01"])

ax.tick_params(axis='both',labelsize=22)
ax.set_ylabel("Precipitación diaria [mm]", fontsize= 26)
ax.tick_params(axis='both',labelsize=24)
ax.set_xlabel("Años", fontsize= 26)
ax.tick_params(axis='both',labelsize=24)
ax.set_yticks(range(10, 130, 10))

plt.setp(ax.get_xticklabels(), rotation = 90)     

# Define the date format
date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.show()
#%% Plotting SERIE COMPLETA en tres TRAMOS
'''Para visulizarla mejor se va a partir la serie total en 3'''

primer_tramo=f_solo_lluvia.loc['1961-01-02':'1981-09-27'] #7574 datos
segundo_tramo= f_solo_lluvia.loc['1981-09-28':'2002-06-23'] #7574 datos
tercer_tramo= f_solo_lluvia.loc['2002-06-24':'2023-03-20'] #7575 datos

#plotting tramos prueba

plt.style.use("ggplot")

fig, axes = plt.subplots(figsize = (40,23), nrows=3, ncols=1, sharey=True, dpi= 100, facecolor='silver')
fig.suptitle('Ezeiza SMN - Precipitación diaria - Enero 1961-Marzo 2023\n   ', fontsize= 33)

#add DataFrames to subplots
axes[0].fill_between(primer_tramo.index, primer_tramo['prcp'], color='blue', linewidth=0.9)
axes[1].fill_between(segundo_tramo.index, segundo_tramo['prcp'], color='blue', linewidth=0.9)
axes[2].fill_between(tercer_tramo.index, tercer_tramo['prcp'], color='blue', linewidth=0.9)

# Axis labels
axes[0].set_xlabel(" ")
axes[0].tick_params(axis='both',labelsize=26)
axes[1].set_xlabel(" ")
axes[1].set_ylabel("Precipitación [mm]", fontsize= 26)
axes[1].tick_params(axis='both',labelsize=26)
axes[2].set_xlabel("Años", fontsize= 28)
axes[2].tick_params(axis='both',labelsize=26)

axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=24))

# Define the date format
date_form = DateFormatter("%Y")
axes[0].xaxis.set_major_formatter(date_form)
axes[1].xaxis.set_major_formatter(date_form)
axes[2].xaxis.set_major_formatter(date_form)
'''
axes[0].xaxis.set_major_locator(MultipleLocator(730))
axes[0].xaxis.set_minor_locator(MultipleLocator(730))
axes[0].set(xlim=["1961", "1981"])
axes[1].xaxis.set_major_locator(MultipleLocator(730))
axes[1].xaxis.set_minor_locator(MultipleLocator(730))
axes[1].set(xlim=["1982", "2002"])
axes[2].xaxis.set_major_locator(MultipleLocator(730))
axes[2].xaxis.set_minor_locator(MultipleLocator(730))
axes[2].set(xlim=["2003", "2023"])
'''

#plt.setp(axes[0].get_xticklabels(), rotation = 90)    
axes[0].set_yticks(range(0, 126, 25))

plt.tight_layout()
plt.savefig("plot/Presentación_de_datos_precip_diaria.png", dpi=300)
plt.show()
#queda acomodaa titulos, ylabel, mejorar visualizacion

#%% SELECCIONO PERIODO 1991-2022 PARA CLIMATOLOGIA
'''
PERCENTILES - CONTROL DE CALIDAD - Desde aca uso la CLIMATOLOGIA

'''
#me fijo que no haya valores negativos
(df['prcp']<0).sum()

#para la CLIMATOLOGIA SELECCIONO UN INTERVALO, el mas actual
df_climatologia=df.loc['1991-01-02':'2021-01-01'] #30 años eL 01 DE ENERO COMO ES 12 UTC TIENE EL DATO CORRESPONDIENTE AL 31 DE DIC DE 2020
df_climatologia.info() #--->10958 de los cuales 14 NAN's y 10944 no nulos
'''
DatetimeIndex: 10958 entries, 1991-01-02 12:00:00 to 2021-01-01 12:00:00
'''
nan_rows_clima = (df_climatologia[df_climatologia.isnull().any(1)]) 
print(nan_rows_clima) #14 Nan's

#%% solo dias con lluvia para PERCENTILES 1991-2022 Y TMB PARA EL 1961-1990
'''PARA PERCENTILES VOY A CONSIDERAR SOLO LOS DIAS CON LLUVIA
o se precipitración diaria distinta de cero'''

f_dfclima = df_climatologia[df_climatologia['prcp']!=0]  #2666 rows
#estadisticos, percentiles, sd... CLIMATOLOGIA COMPLETA
f_dfclima['prcp'].describe().round(1)
'''
Out[14]: 
count    2652.0
mean       11.5
std        15.4
min         0.1
25%         1.0
50%         5.2
75%        15.0
max       120.3
Name: prcp, dtype: float64
'''
f_dfclima['prcp'].quantile([0.9, 0.95, 0.99]).round(1)
'''
Out[15]: 
0.90    31.0
0.95    43.4
0.99    69.7
Name: prcp, dtype: float64
'''
f_dfclima['prcp'].loc[f_dfclima['prcp']==120.3]  #--> 2010-05-24 12utc

############ DESCRIBE PARA PERIODO ANTERIOR ASÍ COMPARO #################

df_clima_anterior= df.loc['1961-01-02':'1991-01-01'] #30 años eL 01 DE ENERO COMO ES 12 UTC TIENE EL DATO CORRESPONDIENTE AL 31 DE DIC DE 2020
df_clima_anterior.info() #--->10957 de los cuales 6 NAN's y 10951 no nulos
'''
DatetimeIndex: 10957 entries, 1961-01-02 12:00:00 to 1991-01-01 12:00:00
'''
nan_rows_clima_anterior = (df_clima_anterior[df_clima_anterior.isnull().any(1)]) 
print(nan_rows_clima_anterior) #6Nan's

'''
                     prcp
Fecha                    
1962-02-25 12:00:00   NaN
1962-03-01 12:00:00   NaN
1964-09-11 12:00:00   NaN
1964-09-12 12:00:00   NaN
1964-10-13 12:00:00   NaN
1964-10-14 12:00:00   NaN
'''
#estadisticos, percentiles, sd...
#solo dias con lluvia
f_dfclima_anterior=df_clima_anterior[df_clima_anterior['prcp']!=0]  #2768 rows
f_dfclima_anterior['prcp'].describe().round(1)
'''
count    2762.0
mean       10.7
std        15.2
min         0.1
25%         1.1
50%         4.7
75%        14.3
max       128.0
Name: prcp, dtype: float64'''

f_dfclima_anterior['prcp'].quantile([0.9, 0.95, 0.99]).round(1)
'''
Out[33]: 
0.90    28.6
0.95    40.8
0.99    73.3
Name: prcp, dtype: float64
'''
f_dfclima_anterior['prcp'].loc[f_dfclima_anterior['prcp']==128]  #--> 1961-12-16 12utc
#%% estadisticos por TRIMESTRE
'''
Estadísticos para el periodo climatologico actual, 
difernciando trimestralmente.

'''
#quiero hacer el describe diferenciando la onda anual

#promedio trimestral de precipitacion diaria (unico)
f_dfclima['Fecha']=pd.to_datetime(f_dfclima.index, format ='%Y/%m/%d')
promedio_trimestral=f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).mean().round(1)
estadistico_trimestral=f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).agg(['mean',
                                                                             'std', 'min', 'max']).round(1)
percentiles_trimestral= f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)

#hago un df con valores estadisticos para cada estacion y luego grafico
#creo un diccionario con los datos de las columnas

data={'Trimestres': ['EFM', 'AMJ', 'JAS', 'OND'], 'media': [13.9, 10.2, 9.2, 12.1], 'Std': [17.1, 14.8, 12.5, 15.9],
      'P75': [19, 13.5, 12, 16.1], 'P95': [52.4, 39.9, 33.8, 45.2],
      'Máx': [105, 120.3, 74, 101]}

climatologia_por_trimestre= pd.DataFrame(data)
climatologia_por_trimestre=climatologia_por_trimestre.set_index('Trimestres')

#plotting
columnas=climatologia_por_trimestre.columns
num_barras = len(columnas)
ancho_barras = 0.8
indice=climatologia_por_trimestre.index
colores = ['#AAA662','#C79FEF', '#7BC8F6', '#76FF7B', '#C875C4']
climatologia_por_trimestre.plot(kind='bar', figsize=(10,4),  color= colores)
# Configurar el título y las etiquetas de los ejes
plt.title('Ezeiza SMN- Climatología trimestral- 1991-2020', fontsize=12)
plt.xlabel('Trimestres', fontsize= 14)
plt.ylabel('Precipitación diaria (mm)', fontsize= 11)
# Mostrar la leyenda
plt.legend(columnas, fontsize=9)
plt.savefig("plot/Climatologia_por_trimestres.png", dpi=200)
plt.show()
#%% PRIMER GRAF DE INTENSIDAD TRIMESTRAL
'''promedio de pp diaria para cada trimestre INTENSIDAD'''
#RESAMPLE.MEAN CALCULA LA MEDIA DE LOS GRUPOS, EXCLUYENDO VALOR FALTANTE
#---> MI GRUPO ES UN TRIMESTRE, ENTONCES AL ACUM DE 90 DIAS LO DIVIDE POR 
# LA CANT DE DIAS QUE LLOVIO -->INTENSIDAD (mm/dia)
promedios_trim=f_dfclima.resample('Q').mean()

#plotting INTENSIDAD
fig, ax = plt.subplots(figsize=(30, 12), facecolor='lavender')
fig.suptitle('Ezeiza SMN - Intensidad de precipitación trimestral', fontsize= 28)
# Add x-axis and y-axis
ax.plot(promedios_trim.index,
       promedios_trim['prcp'], marker='o',
       color='red')

ax.xaxis.set_major_locator(MultipleLocator(365))
ax.xaxis.set_minor_locator(MultipleLocator(365))

# Set title and labels for axes
ax.set(xlim=["1991", "2021"])

ax.tick_params(axis='both',labelsize=22)
ax.set_xlabel("Años", fontsize= 26)
ax.set_ylabel("Intensidad (mm/día)", fontsize= 26)
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)

plt.setp(ax.get_xticklabels(), rotation =45)  
plt.savefig("plot/Intensidad_trimestral.png", dpi=300)
plt.show()

#%%primeros calculos de promedio movil que estan hashteados (no se calcula)
'''ESTO NO CORRERLO

VOLVIENDO A LA CLIMATOLOGIA ACTUAL, NOS QUEDAMOS SOLO CON VAORES 
MAYORES AL P75 (ANUAL)  QUE SON 15 MM, igual al ser anual no es muy representativo
f_dfclima_p75=f_dfclima[f_dfclima['prcp']>=15]
f_dfclima_p75[['prcp']].plot(figsize=(30, 8), fontsize=12) #sin nan, sin frec diaria, solo datos de pp

f_dfclima_p75=f_dfclima_p75.asfreq("D") #solo para el calculo del pm
#calculo pm trimestral para valores mayores o iguales a P75()
pm_trim_p75=f_dfclima_p75['prcp'].rolling(90, min_periods=4).mean().shift(89)
f_dfclima_p75=f_dfclima[f_dfclima['prcp']>=15] #vuelvo atras para la grafica continua
#f_dfclima_p75[['prcp', 'pm_trimestral_p75']].plot(figsize=(20, 8), fontsize=12)
plt.figure(figsize=(40, 12))
plt.plot(f_dfclima_p75.index, f_dfclima_p75['prcp'], label='precipitacion diaria')
plt.plot(pm_trim_p75, label='promedio movil 90D', color='blue')
plt.title('Precipitación diaria igual o mayor a 15 mm (Percentil 75) y promedio movil de 90 días')
plt.ylabel('Precipitación (mm)', fontsize=14)
plt.xlabel('Años', fontsize=16)
plt.tick_params(axis='both',labelsize=18)
plt.legend(fontsize=14)
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.xlim("1991-01", "2019-12")
locator=ticker.MultipleLocator(base=365)
plt.gca().xaxis.set_major_locator(locator)
plt.xticks(rotation=45)
plt.show()

'''

#PROMEDIO MOVIL ANUAL

'''
f_dfclima_pm=f_dfclima.asfreq("D")
#para los dias con lluvia tendria que dejar la frec diaria y poner esos dias como nan
#o hacerlo desde la indexacion que por lo que estoy pensando seria lo mismo.
Promedio_movil=f_dfclima_pm['prcp'].rolling(360, min_periods=1).mean().shift(1) #con shift(1) toma el promedio de los primeros 12 y lo ubica en el lugar 13
#min_periods= número mín de obs en ventana requeridas para tener un valor; de lo contrario, el resultado es
VER, SIGUE SIENDO RARO, GRAFICAR ENCIMA DE L SERIE Y HCERLO A MANO A VER QUE ESTA  TENDRIA Q HACER EL ACUMULADO MENSUAL???
#plotting
plt.figure(figsize=(22,6))
plt.plot(Promedio_movil.index,Promedio_movil, linewidth=0.8, color='green')
plt.ylabel('Precipitación diaria (mm)', fontsize=14)
plt.title('Promedio movil de 12 meses-precipitación diaria (mm) - Ezeiza Aero', fontsize=16)
plt.show()
'''
#%% PROMEDIO MENSUAL PARA LA SERIE CLIMATOLOGICA DE PP 1991-2020

acumulado_mensual=f_dfclima.resample('M').sum() #acumulados para cada mes
acumulado_mensual['Fecha']=pd.to_datetime(acumulado_mensual.index, format ='%Y/%m/%d')
meses_prom=acumulado_mensual.groupby(acumulado_mensual['Fecha'].dt.month).mean().round(1) #promedia los acum mensuales
#meses_frios=[5, 6, 7, 8, 9]
#meses_calidos=[10, 11, 12, 1, 2, 3, 4]
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Ezeiza SMN - Valor medio de precipitación mensual\n 1991-2020')


clrs = ['lightblue' if (4< x < 10) else 'pink' for x in meses_prom.index ]
prom=sns.barplot(ax=ax, x=meses_prom.index, y=meses_prom['prcp'], palette=clrs, linewidth=5)
prom.set(xlabel ="Meses")
prom.set_yticks(range(1, 125, 20))
prom.set_ylabel("Precipitación acumulada promedio (mm)", fontsize= 10)

#pone etiqueta a cada barra
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig("plot/promedio_mensual_pp.png", dpi=200)
plt.show()

#%% TABLA DE ESTADÍSTICOS DE PRECIPITACION PARA ÉPOCA ESTIVAL E INVERNAL DEL SIS
'''período climatologico 1991-2020'''

f_dfclima['Fecha']= pd.to_datetime(f_dfclima['Fecha']) #estan solo los dias de lluvia

#separo los datos en meses fríos y meses cálidos
meses_sis_frio=f_dfclima[f_dfclima['Fecha'].dt.month.isin([5, 6, 7, 8, 9])] #991 datos
meses_sis_calido=f_dfclima[f_dfclima['Fecha'].dt.month.isin([1, 2, 3, 4, 10, 11, 12])] #1675

estadisticos_sis_frio=meses_sis_frio.agg(['mean', 'std', 'min', 'max']).round(1)
estadisticos_sis_calido=meses_sis_calido.agg(['mean', 'std', 'min', 'max']).round(1)

percentiles_sis_frio=meses_sis_frio.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)
percentiles_sis_calido=meses_sis_calido.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)

data_clima_sis={'epoca_sis':['SIS-ESTIVAL', 'SIS-INVERNAL'], 'Mean':[12.9, 9.1],
                'Std': [16.5, 13], 'P75': [17, 12], 'P90': [34.9, 26], 'P95': [48, 34.9],
                'P99': [72.7, 55.6],
                'Máx': [116, 120.3]}

climatologia_epoca_sis=pd.DataFrame(data_clima_sis)
climatologia_epoca_sis= climatologia_epoca_sis.set_index('epoca_sis')
climatologia_epoca_sis=climatologia_epoca_sis.transpose()
climatologia_epoca_sis.to_excel("plot/tabla_climatologiapp_epoca_sis.xlsx")
#%% PLOT DE LA TABLA DE ESTADISTICOS DE PP PARA EPOCA ESTIVAL E INVERNAL DEL SIS
sns.set(font_scale=1.0, style="whitegrid")
#plt.style.use("ggplot")
columnas=climatologia_epoca_sis.columns
num_barras= len(columnas)
ancho_barras=0.8
indice=climatologia_epoca_sis.index
colores=['pink', 'lightblue']
#colores=['#C79FEF', '#7BC8F6']
climatologia_epoca_sis.plot(kind='bar', figsize=(12, 4), color=colores)

#configurar l titulo y las etiquetas de los ejes
plt.title('Climatología de precipitación diaria para la época estival e invernal del patrón SIS\n Período 1991-2020',
          fontsize=11)
plt.xlabel('Estadísticos', fontsize=10)
plt.ylabel('Precipitación diaria (mm)', fontsize=10)
#Mostrar la leyenda
plt.legend(columnas, fontsize=9)
plt.xticks(rotation=0)
plt.savefig('plot/Climatologia_pp_epoca_sis.png', dpi=200)
plt.show()
#%% PERCENTIL 75 PARA CADA MES DEL PERIODO CLIMATOLOGICO (OBVIO SIN LLUVIA CERO)
p75= f_dfclima.groupby(f_dfclima['Fecha'].dt.month).quantile(0.75).round(1)
p75=list(p75['prcp'])
print(p75)
filtro_datos_p75=[]
meses=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in range(len(meses)): 
    filtro_datos_p75.append(f_dfclima[ (f_dfclima.index.month==meses[i]) & (f_dfclima.prcp>=p75[i]) ])
     
filtro_datos_p75= pd.concat(filtro_datos_p75)
print(filtro_datos_p75.index)
filtro_datos_p75=filtro_datos_p75.sort_index()

#filtro_datos_p75['prcp'].plot(kind='bar', width=0.8, figsize=(50,12))
filtro_datos_p75=filtro_datos_p75.asfreq('D')
filtro_datos_p75.index= pd.to_datetime(filtro_datos_p75.index)
#%% plotting Precipitacion SUPERIOR A p75 EN TRES TRAMOS PARA PERIODO CLIMATOLOGICO
filtro_datos_p75['mes']=filtro_datos_p75['Fecha'].dt.month

filtro_1=filtro_datos_p75.loc['1991-01-21':'2001-01-01'] #3634 dttos
filtro_2=filtro_datos_p75.loc['2001-01-02':'2010-12-14'] # 3634 datos
filtro_3=filtro_datos_p75.loc['2010-12-15':'2020-11-25'] #3634 datos
#pruebo plotting diferenciando segun sis por colores
filtro_1frio=filtro_1[filtro_1['Fecha'].dt.month.isin([5, 6, 7, 8, 9])] 
filtro_1calido=filtro_1[filtro_1['Fecha'].dt.month.isin([1, 2, 3, 4, 10, 11, 12])] 

filtro_2frio=filtro_2[filtro_2['Fecha'].dt.month.isin([5, 6, 7, 8, 9])] 
filtro_2calido=filtro_2[filtro_2['Fecha'].dt.month.isin([1, 2, 3, 4, 10, 11, 12])] 

filtro_3frio=filtro_3[filtro_3['Fecha'].dt.month.isin([5, 6, 7, 8, 9])] 
filtro_3calido=filtro_3[filtro_3['Fecha'].dt.month.isin([1, 2, 3, 4, 10, 11, 12])] 

filtro_1frio=filtro_1frio.asfreq('D')
filtro_1calido=filtro_1calido.asfreq('D')
filtro_2frio=filtro_2frio.asfreq('D')
filtro_2calido=filtro_2calido.asfreq('D')
filtro_3frio=filtro_3frio.asfreq('D')
filtro_3calido=filtro_3calido.asfreq('D')

#plt.style.use("ggplot")
sns.set(font_scale=1.0, style="whitegrid")
fig, axes= plt.subplots(figsize=(34, 16), nrows=3, ncols=1, sharey=True,
                       dpi=100, facecolor='#E6E6FA')
fig.suptitle('Extremos de precipitación - Ezeiza SMN \n Período 1991 - 2020', fontsize= 32)
#axes 0
axes[0].fill_between(x=filtro_1frio.index, y1=0, y2=filtro_1frio['prcp'],
                   color='royalblue', linewidth=3.5)
axes[0].fill_between(x=filtro_1calido.index, y1=0, y2=filtro_1calido['prcp'], 
                   color='lightcoral', linewidth=3.5)
axes[0].plot(filtro_1frio.index, filtro_1frio['prcp'], 'o', color='blue', markersize=7, alpha=0.2)
axes[0].plot(filtro_1calido.index, filtro_1calido['prcp'], 'o', color='red', markersize=7, alpha=0.2)

#axes 1
axes[1].fill_between(x=filtro_2frio.index, y1=0, y2=filtro_2frio['prcp'],
                   color='royalblue', linewidth=3.5)
axes[1].fill_between(x=filtro_2calido.index, y1=0, y2=filtro_2calido['prcp'], 
                   color='lightcoral', linewidth=3.5)
axes[1].plot(filtro_2frio.index, filtro_2frio['prcp'], 'o', color='blue', markersize=7, alpha=0.2)
axes[1].plot(filtro_2calido.index, filtro_2calido['prcp'], 'o', color='red', markersize=7, alpha=0.2)

#axes 2
axes[2].fill_between(x=filtro_3frio.index, y1=0, y2=filtro_3frio['prcp'],
                   color='royalblue', linewidth=3.5)
axes[2].fill_between(x=filtro_3calido.index, y1=0, y2=filtro_3calido['prcp'], 
                   color='lightcoral', linewidth=3.5)
axes[2].plot(filtro_3frio.index, filtro_3frio['prcp'], 'o', color='blue', markersize=7, alpha=0.2)
axes[2].plot(filtro_3calido.index, filtro_3calido['prcp'], 'o', color='red', markersize=7, alpha=0.2)

# Axis labels
axes[0].set_xlabel(" ")
axes[0].tick_params(axis='both',labelsize=20)
axes[1].set_xlabel(" ")
axes[1].set_ylabel("Precipitación mayor al P75 [mm]", fontsize= 26)
axes[1].tick_params(axis='both',labelsize=20)
axes[2].set_xlabel("Años", fontsize= 28)
axes[2].tick_params(axis='both',labelsize=20)

axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=12))

# Define the date format
date_form = DateFormatter("%Y-%m")
axes[0].xaxis.set_major_formatter(date_form)
axes[1].xaxis.set_major_formatter(date_form)
axes[2].xaxis.set_major_formatter(date_form)

#plt.setp(axes[0].get_xticklabels(), rotation = 90)    
axes[0].set_yticks(range(0, 126, 25))
# Set title and labels for axes
axes[0].set(xlim=["1991", "2001"])
axes[1].set(xlim=["2001", "2011"])
axes[2].set(xlim=["2011", "2021"])
plt.tight_layout()
plt.savefig("plot/Presentación_de_datos_precip_superior_p75.png", dpi=300)

plt.show()
#%% promedio de pp diaria para cada mes: INTENSIDAD (solo P75) y plot #359datos
#RESAMPLE.MEAN CALCULA LA MEDIA DE LOS GRUPOS, EXCLUYENDO VALOR FALTANTE
#---> MI GRUPO ES UN MES, ENTONCES AL ACUM DE 30 DIAS LO DIVIDE POR 
# LA CANT DE DIAS QUE LLOVIO -->INTENSIDAD (mm/dia)
intensidad_mensual=filtro_datos_p75.resample('M').mean().round(1)
# Diccionario de mapeo de letras por mes
mes_a_letra = {1: 'c', 2: 'c', 3: 'c', 4: 'c', 5: 'f', 6: 'f', 
               7: 'f', 8: 'f', 9: 'f', 10: 'c', 11: 'c', 12: 'c'}

# Agregar una nueva columna 'Periodo' al DataFrame utilizando el mapeo
intensidad_mensual['periodo'] = intensidad_mensual.index.month.map(mes_a_letra)
new_labels = ['SIS-ESTIVAL' if p == 'c' else 'SIS-INVERNAL' for p in intensidad_mensual.periodo]
intensidad_mensual['new_labels']=new_labels
intensidad_mensual['periodo']=intensidad_mensual['new_labels']
intensidad_mensual = intensidad_mensual.drop(columns=['new_labels'])
#colors= ['royalblue' if (p== 'f') else 'lightcoral' for p in intensidad_mensual.periodo]

fig, ax = plt.subplots(figsize=(50,14), facecolor='#E6E6FA')
sns.set_style('white')
sns.set_palette(["red", "green"])
ax=sns.barplot(x=intensidad_mensual.index.strftime('%Y-%m'), y='prcp',
               data=intensidad_mensual, hue='periodo')

plt.title('Intensidad diaria de precipitación para cada mes\nPeríodo 1991-2020', fontsize= 28)
#plotting INTENSIDAD

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.tick_params(axis='both',labelsize=22)
ax.set_xlabel("Años", fontsize= 26)
ax.set_ylabel("Intensidad (mm/día)", fontsize= 26)
ax.grid(True) 


ax.legend(loc='upper left', prop={'size': 20})

plt.savefig("plot/Intensidad_mensual.png", dpi=300)
plt.show()

#%% INTENSIDAD MENSUAL ESTADISTICOS Y BOXPLOT
intensidad_mensual['prcp'].describe().round(1)
intensidad_mensual['prcp'].quantile([0.9, 0.95, 0.99]).round(1)
estad_int_mensual=intensidad_mensual.groupby(intensidad_mensual.index.month).describe().round(1)
estad_int_mensual2=intensidad_mensual.groupby(intensidad_mensual.index.month).quantile([0.9, 0.95, 0.99]).round(1)

intensidad_mensual['mes'] = intensidad_mensual.index.month 
   
fig, ax = plt.subplots(figsize=(16,8), dpi=120)
colores=['lightblue' if (4<x<10) else 'pink' for x in intensidad_mensual.mes]
plt.title("Ezeiza SMN - Intensidad diaria de precipitación\nPeríodo 1991-2020", fontweight= 'bold', fontsize=16)
sns.boxplot(data=intensidad_mensual, x='mes', y='prcp', ax=ax, linewidth= 1.7, palette=colores,
            flierprops={"marker": "x"})
plt.ylabel("Intensidad mensual (mm/día)", fontsize=16)
plt.savefig('plot/boxplot_intens_mensual.png', dpi=300)
ax.tick_params(axis='both',labelsize=14)
plt.show()
#%% filtro PRECIPITACION EN PERIODO SIS: 1979-2016 (SOLO p75)
#voy  a calcular intensidd mensual para valores de pp diaria superiores al p75
#pero ahora considerando el periodo para el que tengo datos sis asi luego comparar

lluvia_periodosis=df.loc['1979-10-01':'2016-04-30']
#me quedare solo con dias e lluvia distinta de cero para calculos estadisticos
filtro_lluvia_periodosis=lluvia_periodosis[lluvia_periodosis['prcp']!=0]
#necesitare poner el indice fecha como columna para el prox calulo del percentil  75 a nivel mensual
filtro_lluvia_periodosis['Fecha']=filtro_lluvia_periodosis.index
per75lluvia_periodosis=filtro_lluvia_periodosis.groupby(filtro_lluvia_periodosis['Fecha'].dt.month).quantile(0.75).round(1)
per75lluvia_periodosis=list(per75lluvia_periodosis['prcp'])

print(per75lluvia_periodosis)
filtro_datos_p75_periodosis=[]
meses=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in range(len(meses)): 
    filtro_datos_p75_periodosis.append(filtro_lluvia_periodosis[(filtro_lluvia_periodosis.index.month==meses[i]) 
                                                                & (filtro_lluvia_periodosis.prcp>=per75lluvia_periodosis[i])])
     
filtro_datos_p75_periodosis= pd.concat(filtro_datos_p75_periodosis)
print(filtro_datos_p75_periodosis.index)
filtro_datos_p75_periodosis=filtro_datos_p75_periodosis.sort_index()

#filtro_datos_p75['prcp'].plot(kind='bar', width=0.8, figsize=(50,12))
filtro_datos_p75_periodosis=filtro_datos_p75_periodosis.asfreq('D')
filtro_datos_p75_periodosis.index= pd.to_datetime(filtro_datos_p75_periodosis.index)
#%% INTENSIDAD MENSUAL PARA ESE FILTRADO EN PERIODO SIS Y BARPLOT #439 DATOS
#ahora si puedo calcular la intensidad mensual con mi df 
#de frecuencia diaria que tiene solo los valores ue igualan o superan p75 
#para el periodo coincidente con periodo de base de datos sis

filtro_datos_p75_periodosis['mes']=filtro_datos_p75_periodosis['Fecha'].dt.month

intensidad_mensual_lluvia_periodosis=filtro_datos_p75_periodosis.resample('M').mean().round(1)
# Diccionario de mapeo de letras por mes
mes_a_letra = {1: 'c', 2: 'c', 3: 'c', 4: 'c', 5: 'f', 6: 'f', 
               7: 'f', 8: 'f', 9: 'f', 10: 'c', 11: 'c', 12: 'c'}

# Agregar una nueva columna 'Periodo' al DataFrame utilizando el mapeo
intensidad_mensual_lluvia_periodosis['periodo'] = intensidad_mensual_lluvia_periodosis.index.month.map(mes_a_letra)
new_labels = ['SIS-ESTIVAL' if p == 'c' else 'SIS-INVERNAL' for p in intensidad_mensual_lluvia_periodosis.periodo]
intensidad_mensual_lluvia_periodosis['new_labels']=new_labels
intensidad_mensual_lluvia_periodosis['periodo']=intensidad_mensual_lluvia_periodosis['new_labels']
intensidad_mensual_lluvia_periodosis = intensidad_mensual_lluvia_periodosis.drop(columns=['new_labels'])

fig, ax = plt.subplots(figsize=(50,14), facecolor='#E6E6FA')
sns.set_style('white')
sns.set_palette(["red", "green"])
ax=sns.barplot(x=intensidad_mensual_lluvia_periodosis.index.strftime('%Y-%m'), y='prcp',
               data=intensidad_mensual_lluvia_periodosis, hue='periodo')

plt.title('Intensidad diaria de precipitación para valores extremos\n Período 1979-2016', fontsize= 28)
#plotting INTENSIDAD
ax.set_yticks(range(0, 140, 20))
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.tick_params(axis='both',labelsize=22)
ax.set_xlabel("Años", fontsize= 26)
ax.set_ylabel("Intensidad (mm/día)", fontsize= 26)
ax.grid(True) 


ax.legend(loc='upper left', prop={'size': 20})

plt.savefig("plot/Intensidad_mensual_lluviaenperiodosis.png", dpi=300)
plt.show()

#%%INTENSIDAD MENSUAL  en periodo SIS: ESTADISTICOS Y BOXPLOT
intensidad_mensual_lluvia_periodosis['prcp'].describe().round(1)
intensidad_mensual_lluvia_periodosis['prcp'].quantile([0.9, 0.95, 0.99]).round(1)
estad_int_mensual_lluviasis=intensidad_mensual_lluvia_periodosis.groupby(
    intensidad_mensual_lluvia_periodosis.index.month).describe().round(1)
estad_int_mensual_lluviasis2=intensidad_mensual_lluvia_periodosis.groupby(
    intensidad_mensual_lluvia_periodosis.index.month).quantile([0.9, 0.95, 0.99]).round(1)

intensidad_mensual_lluvia_periodosis['mes'] = intensidad_mensual_lluvia_periodosis.index.month 
   
fig, ax = plt.subplots(figsize=(16,8), dpi=120)
colores=['lightblue' if (4<x<10) else 'pink' for x in intensidad_mensual.mes]
plt.title("Intensidad diaria de precipitación para valores extremos \n Período 1979-2016", fontweight= 'bold', fontsize=16)
sns.boxplot(data=intensidad_mensual_lluvia_periodosis, x='mes', y='prcp', ax=ax, linewidth= 1.7, palette=colores,
            flierprops={"marker": "x"})
plt.ylabel("Intensidad mensual (mm/día)", fontsize=16)
plt.savefig('plot/boxplot_intens_mensual_lluvia_peridosis.png', dpi=300)
ax.tick_params(axis='both',labelsize=14)
plt.show()
#%% DF CON PP > P75 MENSUAL, PARA LOS MESES FRIOS Y PARA LOS MESES CALIDOS POR SEPARADO

################## FRIO ######################
#OBTENGO UN DF QUE TIENE SOLO LOS MESES DE SIS INVERNAL CON PP > A P75 DE CADA MES
p75_frio=meses_sis_frio.groupby(meses_sis_frio['Fecha'].dt.month).quantile(0.75).round(1)
p75_frio=list(p75_frio['prcp'])
print(p75_frio)
meses_frios=[5, 6, 7, 8, 9]
datos_filtrados_frio=[]     
for i in range(len(meses_frios)):
    datos_filtrados_frio.append(meses_sis_frio[(meses_sis_frio.index.month==meses_frios[i])
                                                   & (meses_sis_frio.prcp>=p75_frio[i])])
 
datos_filtrados_frio= pd.concat(datos_filtrados_frio)
print(datos_filtrados_frio.index)
datos_filtrados_frio=datos_filtrados_frio.sort_index()

################## CALIDO ######################
#OBTENGO UN DF QUE TIENE SOLO LOS MESES DE SIS ESTIVAL CON PP > A P75 DE CADA MES
p75_calido=meses_sis_calido.groupby(meses_sis_calido['Fecha'].dt.month).quantile(0.75).round(1)
p75_calido=list(p75_calido['prcp'])
print(p75_calido)

meses_calidos=[1, 2, 3, 4, 10, 11, 12]
datos_filtrados_calido=[]     
for i in range(len(meses_calidos)):
    datos_filtrados_calido.append(meses_sis_calido[(meses_sis_calido.index.month==meses_calidos[i])
                                                   & (meses_sis_calido.prcp>=p75_calido[i])])
 
datos_filtrados_calido= pd.concat(datos_filtrados_calido)
print(datos_filtrados_calido.index)
datos_filtrados_calido=datos_filtrados_calido.sort_index()
#%% plotting ACUMULADOS MENSUALES  periodo climatologico ----> (NO USADO) agregarle la media mensual
#podri servir para comparar con el promedio mensual de pp acumulada para 
#cada mes, el grafico rosa y celeste de barras. Lo dejamos por las dudas
# nos sirva en algun momento
acumulado_mensual['mes']=acumulado_mensual.index.month
sns.set(font_scale=1.0, style="whitegrid")
#meses['mes']=meses.index.month
fig, ax=plt.subplots(figsize=(30,7))
                                               
ax.plot(acumulado_mensual.index, acumulado_mensual.prcp,
        marker='o', color='b', linestyle='--')

ax.set_xlabel("Meses", fontsize=22)
ax.set_ylabel("Precipitación mensual (mm)", fontsize=22)
ax.set_title("Ezeiza SMN- Precipitación acumulada mensual\n1991-2020", fontsize=24)

ax.grid(True)
ax.tick_params(axis='both',labelsize=18) #solo tamaño de_los_ejes, no sus leyendas
date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(ticker.MultipleLocator(730))
ax.set(xlim=["1990-12", "2021-12"])
#%%PP TRIMESTRAL ACUMULADA VS EL PROMEDIO DE PP TRIMESTRAL PARA CADA ESTACION (NO USADO)
#Nos podria servir para ver que veanos fueron mas lluviosos, por ejemplo, si estu
#vieron muy por encima de la media de los acuulados de pp del verano.
#igual tiene errores en el ultimo subplot y falta mejorar el grafico.
acum_trimestral=f_dfclima.resample('Q').sum()
acum_trimestral['Fecha'] =pd.to_datetime(acum_trimestral.index,format ='%Y/%m/%d')
mean_acum_trimestral=acum_trimestral.groupby(acum_trimestral['Fecha'].dt.quarter).mean().round(1)
mean_acum_trimestral=mean_acum_trimestral['prcp'].tolist()
std_acum_trimestral= acum_trimestral.groupby(acum_trimestral['Fecha'].dt.quarter).std().round(1)
print(mean_acum_trimestral)
'''
        prcp
Fecha       
1      312.3
2      217.9
3      178.5
4      304.7
'''
print(std_acum_trimestral)
'''
        prcp
Fecha       
1      132.8
2       98.6
3       71.2
4      121.8'''

#plotting
acum_trimestral['mes']=acum_trimestral.index.month
fig, ax=plt.subplots(4,1,sharey=True, figsize=(20,10))
                                               
ax[0].plot(acum_trimestral[acum_trimestral.mes==3].index, 
           acum_trimestral[acum_trimestral.mes==3].prcp,
           marker='o', color='b', linestyle='--', label='veranos')
ax[0].axhline(y=mean_acum_trimestral[0],
              color='r', linestyle='--', label='media verano')

ax[1].plot(acum_trimestral[acum_trimestral.mes==6].index, 
           acum_trimestral[acum_trimestral.mes==6].prcp,
           marker='o', color='b', linestyle='--', label='otoños')
ax[1].axhline(y=mean_acum_trimestral[1],
              color='r', linestyle='--', label='media otoño')

ax[2].plot(acum_trimestral[acum_trimestral.mes==9].index,
           acum_trimestral[acum_trimestral.mes==9].prcp,
           marker='o', color='b', linestyle='--', label='inviernos')
ax[2].axhline(y=mean_acum_trimestral[2],
              color='r', linestyle='--', label='media invierno')

ax[3].plot(acum_trimestral[acum_trimestral.mes==12].index, 
           acum_trimestral[acum_trimestral.mes==9].prcp,
           marker='o', color='b', linestyle='--', label='primaveras')
ax[3].axhline(y=mean_acum_trimestral[3],
              color='r', linestyle='--', label='media primavera')

ax[3].set_xlabel('años')
#ax[0].set_ylabel('Precipitación acumulada (mm)')
#ax[1].set_ylabel('Precipitación acumulada (mm)')
ax[2].set_ylabel('Precipitación acumulada (mm)')
#ax[3].set_ylabel('Precipitación acumulada (mm)')
ax[0].set_title('Precipitación trimestral vs. su valor medio (mm)-Ezeiza Aero')
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
plt.show()
#%% ABRO Y ACOMODO SERIE DE DATOS DIARIOS INDICE SIS DE OCTUBRE A ABRIL #7844 datos
'''LEO Y ABRO ARCHIVOS CORRESPONDIENTES A VARIABLE SIS OA
Y LOS ACOMODO QUEDANDOME UN DATAFRAME CON INDICE FECHA Y COLUMNA DE VALORES SIS
'''
directorio = 'data_set'
archivo2_sis_OA= 'SIS7914_ONDEFMA1090.csv'
fname2 = os.path.join(directorio,archivo2_sis_OA)
dfSIS_OA= pd.read_csv(fname2, sep= ',', encoding='latin-1') #/216, 1025)

'''ELIMINO LAS FILAS Y COLUMNAS QUE NO NECESITO
PARA EL "dfSIS_OA"
'''
dfSIS_OA= dfSIS_OA.drop(dfSIS_OA.columns[38: 1025], axis=1)
dfSIS_OA= dfSIS_OA.drop(dfSIS_OA.index[0:3], axis=0)
dfSIS_OA= dfSIS_OA.drop(dfSIS_OA.index[212], axis=0)
dfSIS_OA.columns
dfSIS_OA = dfSIS_OA.rename(columns={'Índice SIS 10-90 para la estación Octubre-Abril*       (*) Se indica el año de comienzo de la estación':'meses'})
#creo un indice para el rango de datos, que exceptue ciertos meses del año
def crear_indice_fechas(inicio, fin, meses_a_excluir):
    # Generar todas las fechas desde inicio hasta fin
    rango_fechas = pd.date_range(start=inicio, end=fin, freq='D')

    # Filtrar las fechas excluyendo los meses especificados
    indice_fechas = [fecha for fecha in rango_fechas if fecha.month not in meses_a_excluir]
  
    return indice_fechas

# Definir el rango de fechas deseado (desde el 1 de octubre de 1979 hasta el 30 de abril de 2016)
fecha_inicio = '1979-10-01'
fecha_fin = '2016-04-30'
# Definir los meses que deseamos excluir (en este caso Mayo a Septiembre)
meses_excluir = {5, 6, 7, 8, 9}  # Usamos un conjunto para acelerar las comprobaciones

# Crear el índice de fechas
indice_fechas = crear_indice_fechas(fecha_inicio, fecha_fin, meses_excluir)
#elimino los 29 de febrero porque no estan en la base de datos sis
for fecha in indice_fechas:
    if (fecha.month==2) & (fecha.day==29):
        indice_fechas.remove(fecha)

# Unificar todas las columnas del dfSIS_OA en una columna "meses" y otra columna "Valor"
dfSIS_OA.size-212 #7844 porque le reste 212 que es la col de meses
dfSIS_OA= pd.melt(dfSIS_OA, id_vars=['meses'], var_name='Año', value_name='Valor')
dfSIS_OA= dfSIS_OA.drop(['Año'], axis=1)

#necesito poner como indice a la lista de fechas que cree 
dfSIS_OA.index=indice_fechas
#luego de constatar que coincida la columna meses con el indice fechas, la elimino
dfSIS_OA= dfSIS_OA.drop(['meses'], axis=1)
#%% Plotting primera vista de los datos sis_OA
fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_OA, palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 1979-2016 considerando solo meses de Octubre a Abril',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)

fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_OA.loc['2012':'2016'], palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 2012-2016 considerando solo meses de Octubre a Abril',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)
#%% CLIMATOLOGIA PARA VALORES POSITIVOS Y VALORES NEGATIVOS DEL SIS_OA (tabla combined_describe_sisOA)
# Verificar si hay algún valor cero en el DataFrame
hay_valor_cero = (dfSIS_OA == 0).any().any()
if hay_valor_cero:
    print("Hay al menos un valor cero en el DataFrame.")
else:
    print("No hay valores cero en el DataFrame.")
#RETURN: No hay valores ceros en el Dataframe

# FILTRAR VALORES POSITIVOS
valores_positivos_sis_OA = dfSIS_OA[dfSIS_OA['Valor'] > 0]['Valor']
valores_positivos_sis_OA =pd.DataFrame(valores_positivos_sis_OA)
print("Valores positivos en la columna {}:\n{}".format('Valor', valores_positivos_sis_OA)) #3948 datos positivos
describe_val_pos_OA=valores_positivos_sis_OA.describe().round(2)
percentiles_valores_positivos_sis_OA=valores_positivos_sis_OA.quantile([0.9, 0.95, 0.99]).round(2)
describe_val_pos_OA=describe_val_pos_OA.append(percentiles_valores_positivos_sis_OA)
describe_val_pos_OA=describe_val_pos_OA.rename(columns={'Valor':'SIS-ESTIVAL-Positivo'})
describe_val_pos_OA.to_excel('plot/descripcion_valores_pos_OA.xlsx', index=True)


# FILTRAR VALORES NEGATIVOS
valores_negativos_sis_OA = dfSIS_OA[dfSIS_OA['Valor'] < 0]['Valor']
valores_negativos_sis_OA =pd.DataFrame(valores_negativos_sis_OA)
print("Valores negativos en la columna {}:\n{}".format('Valor', valores_negativos_sis_OA)) #3896 datos negativos
#los voy a multiplicar por (-1) para calculos estadisticos, mas q nada de los percentiles
valores_negativos_sis_OA['Valor']= valores_negativos_sis_OA['Valor'] * (-1)
describe_val_neg_OA=valores_negativos_sis_OA.describe().round(2)
percentiles_valores_negativos_sis_OA=valores_negativos_sis_OA.quantile([0.9, 0.95, 0.99]).round(2)
describe_val_neg_OA=describe_val_neg_OA.append(percentiles_valores_negativos_sis_OA)
#vuelvo a multiplicar ahora si por -1
describe_val_neg_OA['Valor']= describe_val_neg_OA['Valor'] * (-1)
describe_val_neg_OA.at['std', 'Valor'] = 0.63
describe_val_neg_OA.at['min', 'Valor'] = 0
describe_val_neg_OA=describe_val_neg_OA.rename(columns={'Valor':'SIS-ESTIVAL-Negativo'})
describe_val_neg_OA.to_excel('plot/descripcion_valores_neg_OA.xlsx', index=True)

#junto las 2 tablas de dataframe de estadisticos. positivas y negativas
combined_describe_sisOA = pd.concat([describe_val_pos_OA, describe_val_neg_OA], axis=1)
combined_describe_sisOA.to_excel('plot/tabla_estadisticos_sisOA_negypos.xlsx', index=True)
#aca deberia graficar como con sis calido y frio
combined_describe_sisOA=combined_describe_sisOA.drop(['count', 'min'], axis=0)
#quiero mover la fila de maximo hacia el final
#la guardo en una variable temporal
fila_temporal=combined_describe_sisOA.iloc[5]
# Elimina la fila del DataFrame original
combined_describe_sisOA = combined_describe_sisOA.drop(combined_describe_sisOA.index[5])
# Inserta la fila en la nueva posición
combined_describe_sisOA = combined_describe_sisOA.append(fila_temporal, ignore_index=True)
nuevos_indices= ['Mean', 'Std', 'P25', 'P50', 'P75', 'P90', 'P95','P99', 'Max']
combined_describe_sisOA=combined_describe_sisOA.rename(index=dict(zip(combined_describe_sisOA.index, nuevos_indices)))
combined_describe_sisOA.to_excel('plot/ESTADISTICOS_OA.xlsx', index=True)
#vuelvo a dejar los valores negativos como negativos ya que hice los calculos
valores_negativos_sis_OA['Valor']= valores_negativos_sis_OA['Valor'] * (-1)
#%%PLOTEO LOS ESTADISTICOS DEL SIS OA PARA VALORES POSITIVOS Y VALORES NEGATIVOS
indice=combined_describe_sisOA.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Estadísticos para los meses de Octubre a Abril del índice SIS\nPeríodo 1979-2016', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2,combined_describe_sisOA["SIS-ESTIVAL-Negativo"], 
       width=ancho_barras, color='dodgerblue', label="SIS-ESTIVAL-Negativo")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_describe_sisOA["SIS-ESTIVAL-Positivo"], width=ancho_barras,
       color='#C79FEF', label="SIS-ESTIVAL-Positivo")

#ax = sns.barplot(data=combined_describe_sisOA, x=indice, y=combined_describe_sisOA["SIS-ESTIVAL-Negativo"], 
#            color='dodgerblue', label="SIS-ESTIVAL-Negativo", ax=ax)
#sns.barplot(data=combined_describe_sisOA, x=indice, y=combined_describe_sisOA["SIS-ESTIVAL-Positivo"],
#                 color='#C79FEF', label="SIS-ESTIVAL-Positivo")


# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")
ax.legend(loc='upper left', fontsize=9)
ax.set_yticks(range(-4, 6, 2))

ax.set_ylabel('Índice SIS', fontsize=11)


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
plt.savefig('plot/Estadisticos_indice_sisOA.png', dpi=200)
plt.show()
#%% PLOTEO LO MISMO PERO EN FORMATO BOXPLOT HORIZONTAL
'''Ahora similar pero con boxplot'''
# Crea una figura y dos ejes, uno para valores positivos y otro para valores negativos
fig, (ax_negativos, ax_positivos) = plt.subplots(1, 2, figsize=(20,6)) #sharex=True para que los subplots compartan el mismo eje x
ax_negativos.set_xlim(-5.5, 0.1)
ax_positivos.set_xlim(-0.1, 5.5)
ax_positivos.grid(True)
ax_negativos.grid(True)
fig.suptitle('Valores positivos y negativos del índice SIS\n Octubre a Abril, Período 1979-2016', fontsize=18)

# Eliminar las líneas de grid horizontales
ax_negativos.yaxis.grid(False)
ax_positivos.yaxis.grid(False)

# Grafica los boxplots para los valores positivos y negativos
ax_negativos.boxplot(valores_negativos_sis_OA , vert=False, flierprops={'marker':'x'}, 
                     showfliers= True, showmeans=True, meanline= True, notch=True, patch_artist=True,
                     boxprops=dict(facecolor='dodgerblue', color='black'),
                     meanprops= dict(linestyle='-', linewidth=1.5, color='green'))
ax_negativos.set_ylabel(' ')
ax_negativos.set_yticklabels([' '])
ax_positivos.boxplot(valores_positivos_sis_OA , vert=False, flierprops={'marker':'x'}, 
                     showfliers= True, showmeans=True, meanline= True, notch=True, patch_artist=True,
                     boxprops=dict(facecolor='#C79FEF', color='black'), 
                     meanprops= dict(linestyle='-', linewidth=1.5, color='green'))

# Establece el título y etiquetas del gráfico
ax_positivos.set_xlabel('Índice SIS-Valores Positivos', fontsize=14)
ax_negativos.set_xlabel('Índice SIS-Valores Negativos', fontsize=14)
ax_positivos.set_ylabel(' ')
ax_positivos.set_yticklabels([' '])
# Muestra el gráfico
plt.tight_layout #asegura que este todo bien ajustado
plt.savefig('plot/boxplot_indice_SIS.png', dpi=300)
plt.show()
#%% ABRO Y ACOMODO SERIE DE DATOS DIARIOS INDICE SIS DE MAYO A SEPTIEMBRE #5508 datos
archivo3_sis_MS= 'SIS8015_MJJAS1090.csv'
fname3 = os.path.join(directorio,archivo3_sis_MS)
dfSIS_MS= pd.read_csv(fname3, sep= ',', encoding='latin-1') #(156, 37)

'''ELIMINO LAS FILAS Y COLUMNAS QUE NO NECESITO
PARA EL "dfSIS_MS"
'''
dfSIS_MS= dfSIS_MS.drop(dfSIS_MS.index[0:3], axis=0)
dfSIS_MS.columns
dfSIS_MS = dfSIS_MS.rename(columns={'Índice SIS 10-90 para la estación Mayo-Septiembre':'meses'})
#creo un indice para el rango de datos, que exceptue ciertos meses del año
'''def crear_indice_fechas(inicio, fin, meses_a_excluir):
    # Generar todas las fechas desde inicio hasta fin
    rango_fechas = pd.date_range(start=inicio, end=fin, freq='D')

    # Filtrar las fechas excluyendo los meses especificados
    indice_fechas = [fecha for fecha in rango_fechas if fecha.month not in meses_a_excluir]
  
    return indice_fechas
'''
# Definir el rango de fechas deseado (desde el 1 de Mayo de 1980 hasta el 30 de Septiembre de 2015)
fecha_inicio_ms = '1980-05-01'
fecha_fin_ms = '2015-09-30'
# Definir los meses que deseamos excluir (en este caso e-f-m-a-o-n-d)
meses_excluir = {1 ,2, 3, 4, 10, 11, 12}  # Usamos un conjunto para acelerar las comprobaciones

# Crear el índice de fechas
indice_fechas_ms = crear_indice_fechas(fecha_inicio_ms, fecha_fin_ms, meses_excluir)

# Unificar todas las columnas del dfSIS_OA en una columna "meses" y otra columna "Valor"
dfSIS_MS.shape #(153,37)
dfSIS_MS.size-153 #5661 porque le reste 153 que es la col de meses-->5508
dfSIS_MS= pd.melt(dfSIS_MS, id_vars=['meses'], var_name='Año', value_name='Valor')
dfSIS_MS= dfSIS_MS.drop(['Año'], axis=1)

#necesito poner como indice a la lista de fechas que cree 
dfSIS_MS.index=indice_fechas_ms
#luego de constatar que coincida la columna meses con el indice fechas, la elimino
dfSIS_MS= dfSIS_MS.drop(['meses'], axis=1)
#%% Plotting primera vista de los datos sis_MS
fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_MS, palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 1980-2015 considerando solo meses de Mayo a Septiembre',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)

fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_MS.loc['2012':'2016'], palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 2012-2016 considerando solo meses de Mayo a Septiembre',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)
#%% CLIMATOLOGIA PARA VALORES POSITIVOS Y VALORES NEGATIVOS DEL SIS_MS (tabla combined_describe_sisMS)
'''
AHORA CALCULO CLIMATOLOGIA SEPARANDO EN VALORES POSITIVOS Y VALORES NEGATIVOS DEL SIS_MS
'''
# Verificar si hay algún valor cero en el DataFrame
hay_valor_cero = (dfSIS_MS == 0).any().any()
if hay_valor_cero:
    print("Hay al menos un valor cero en el DataFrame.")
else:
    print("No hay valores cero en el DataFrame.")

#RETURN: No hay valores ceros en el Dataframe

# FILTRAR VALORES POSITIVOS
valores_positivos_sis_MS = dfSIS_MS[dfSIS_MS['Valor'] > 0]['Valor']
valores_positivos_sis_MS =pd.DataFrame(valores_positivos_sis_MS)
print("Valores positivos en la columna {}:\n{}".format('Valor', valores_positivos_sis_MS.count())) #2722 datos positivos
describe_val_pos_MS=valores_positivos_sis_MS.describe().round(2)
percentiles_valores_positivos_sis_MS=valores_positivos_sis_MS.quantile([0.9, 0.95, 0.99]).round(2)
describe_val_pos_MS=describe_val_pos_MS.append(percentiles_valores_positivos_sis_MS)
describe_val_pos_MS=describe_val_pos_MS.rename(columns={'Valor':'SIS-INVERNAL-Positivo'})
describe_val_pos_MS.to_excel('plot/descripcion_valores_pos_MS.xlsx', index=True)


# FILTRAR VALORES NEGATIVOS
valores_negativos_sis_MS = dfSIS_MS[dfSIS_MS['Valor'] < 0]['Valor']
valores_negativos_sis_MS =pd.DataFrame(valores_negativos_sis_MS)
print("Valores negativos en la columna {}:\n{}".format('Valor', valores_negativos_sis_MS.count())) #2786 datos negativos
#los voy a multiplicar por (-1) para calculos estadisticos, mas q nada de los percentiles
valores_negativos_sis_MS['Valor']= valores_negativos_sis_MS['Valor'] * (-1)
describe_val_neg_MS=valores_negativos_sis_MS.describe().round(2)
percentiles_valores_negativos_sis_MS=valores_negativos_sis_MS.quantile([0.9, 0.95, 0.99]).round(2)
describe_val_neg_MS=describe_val_neg_MS.append(percentiles_valores_negativos_sis_OA)
#vuelvo a multiplicar ahora si por -1
describe_val_neg_MS['Valor']= describe_val_neg_MS['Valor'] * (-1)
describe_val_neg_MS.at['std', 'Valor'] = 0.59
describe_val_neg_MS.at['min', 'Valor'] = 0
describe_val_neg_MS=describe_val_neg_MS.rename(columns={'Valor':'SIS-INVERNAL-Negativo'})
describe_val_neg_MS.to_excel('plot/descripcion_valores_neg_MS.xlsx', index=True)

#junto las 2 tablas de dataframe de estadisticos. positivas y negativas
combined_describe_sisMS = pd.concat([describe_val_pos_MS, describe_val_neg_MS], axis=1)
combined_describe_sisMS.to_excel('plot/tabla_estadisticos_sisMS_negypos.xlsx', index=True)
#aca deberia graficar como con sis calido y frio
combined_describe_sisMS=combined_describe_sisMS.drop(['count', 'min'], axis=0)
#quiero mover la fila de maximo hacia el final
#la guardo en una variable temporal
fila_temporal=combined_describe_sisMS.iloc[5]
# Elimina la fila del DataFrame original
combined_describe_sisMS = combined_describe_sisMS.drop(combined_describe_sisMS.index[5])
# Inserta la fila en la nueva posición
combined_describe_sisMS = combined_describe_sisMS.append(fila_temporal, ignore_index=True)
nuevos_indices= ['Mean', 'Std', 'P25', 'P50', 'P75', 'P90', 'P95','P99', 'Max']
combined_describe_sisMS=combined_describe_sisMS.rename(index=dict(zip(combined_describe_sisMS.index, nuevos_indices)))
combined_describe_sisMS.to_excel('plot/ESTADISTICOS_MS.xlsx', index=True)
#vuelvo a dejar los valores negativos como negativos ya que hice los calculos
valores_negativos_sis_MS['Valor']= valores_negativos_sis_MS['Valor'] * (-1)
#%% PLOTEO LOS ESTADISTICOS DEL SIS OA PARA VALORES POSITIVOS Y VALORES NEGATIVOS
indice=combined_describe_sisMS.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Estadísticos para los meses de Mayo a Septiembre del índice SIS\nPeríodo 1980-2015', fontsize=12)

# Ancho de las barras
ancho_barras = 0.35

# Crear las posiciones x para los grupos de barras
posiciones_x = np.arange(len(indice))

# Trazar las barras del primer DataFrame en la primera posición
ax.bar(posiciones_x - ancho_barras/2,combined_describe_sisMS["SIS-INVERNAL-Negativo"], 
       width=ancho_barras, color='dodgerblue', label="SIS-INVERNAL-Negativo")

# Trazar las barras del segundo DataFrame en la segunda posición
ax.bar(posiciones_x + ancho_barras/2, combined_describe_sisMS["SIS-INVERNAL-Positivo"], width=ancho_barras,
       color='#C79FEF', label="SIS-INVERNAL-Positivo")
# Configurar etiquetas del eje x con los índices del DataFrame
ax.set_xticks(posiciones_x)
ax.set_xticklabels(indice, rotation=0, ha="right")

ax.legend(loc='upper left', fontsize=9)
ax.set_yticks(range(-4, 6, 2))
ax.set_ylabel('Índice SIS', fontsize=11)
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

plt.savefig('plot/Estadisticos_indice_sisMS.png', dpi=200)
plt.show()
#%% PLOTEO LO MISMO PERO EN FORMATO BOXPLOT HORIZONTAL (MS)
'''Ahora similar pero con boxplot'''

# Crea una figura y dos ejes, uno para valores positivos y otro para valores negativos
fig, (ax_negativos, ax_positivos) = plt.subplots(1, 2, figsize=(20,6))
ax_negativos.set_xlim(-5.5, 0.1)
ax_positivos.set_xlim(-0.1, 5.5)
ax_positivos.grid(True)
ax_negativos.grid(True)
fig.suptitle('Valores positivos y negativos del índice SIS\n Mayo a Septiembre, Período 1980-2015', fontsize=18)

# Eliminar las líneas de grid horizontales
ax_negativos.yaxis.grid(False)
ax_positivos.yaxis.grid(False)

# Grafica los boxplots para los valores positivos y negativos
ax_negativos.boxplot(valores_negativos_sis_MS , vert=False, flierprops={'marker':'x'}, 
                     showfliers= True, showmeans=True, meanline= True, notch=True, patch_artist=True,
                     boxprops=dict(facecolor='dodgerblue', color='black'),
                     meanprops= dict(linestyle='-', linewidth=1.5, color='green'))
                 
ax_negativos.set_ylabel(' ')
ax_negativos.set_yticklabels([' '])
ax_positivos.boxplot(valores_positivos_sis_MS , vert=False, flierprops={'marker':'x'}, 
                     showfliers= True, showmeans=True, meanline= True, notch=True, patch_artist=True,
                     boxprops=dict(facecolor='#C79FEF', color='black'), 
                     meanprops= dict(linestyle='-', linewidth=1.5, color='green'))

# Establece el título y etiquetas del gráfico

ax_positivos.set_xlabel('Índice SIS-Valores Positivos', fontsize=14)
ax_negativos.set_xlabel('Índice SIS-Valores Negativos', fontsize=14)
ax_positivos.set_ylabel(' ')
ax_positivos.set_yticklabels([' '])
# Muestra el gráfico
plt.tight_layout() #asegura que este todo bien ajustado
plt.savefig('plot/boxplot_indice_SIS_MS.png', dpi=300)
plt.show()
#%%CONCATENO LOS DATAFRAMES DEL INDICE SIS DE AMBOS PERIODOS CALIDO Y FRIO Y LOS ORDENO POR INDICE #13352datos
# Concatenar los DataFrames y ordenar por índice
SIS_completo = pd.concat([dfSIS_OA, dfSIS_MS])
SIS_completo = SIS_completo.sort_index()
# Agregar una columna para el mes
SIS_completo['mes'] = SIS_completo.index.month
mes_a_letra = {1: 'c', 2: 'c', 3: 'c', 4: 'c', 5: 'f', 6: 'f', 
               7: 'f', 8: 'f', 9: 'f', 10: 'c', 11: 'c', 12: 'c'}
''' SIS_completo ARRANCA EL 01/10/1979 y finaliza el 30-04-2016
'''
# Agregar una nueva columna 'Periodo' al DataFrame utilizando el mapeo
SIS_completo['periodo'] = SIS_completo.index.month.map(mes_a_letra)
#%% cuento cantidad de sis calidos y frios en toda la serie para ver si coinciden
conteo_categorias = SIS_completo['periodo'].value_counts()
# Imprime el resultado
print(conteo_categorias)
#result
'''
c    7844
f    5508
'''
#periodo frio dura 5 meses (153 dias tiene cada periodo frio de mayo a septiembre )--> 5508/153 = 36 PERÍODOS FRÍOS
#periodo calido (212 dias tiene cada periodo de octubre a abril, sin contar los años bisiestos que tendrian un dia mas y son 10 pero aca no estan)
# --> 7844/212 = 37 PERÍODOS CÁLIDOS
#%% DIVIDO LA SERIE COMPLETA DE SIS EN TRES PERIODOS PARA GRAFICAR
primer_tramo_sis=SIS_completo.loc['1979-10-01':'1991-09-30'] #4380 datos
segundo_tramo_sis= SIS_completo.loc['1991-10-01':'2003-09-30'] #4380 datos
tercer_tramo_sis= SIS_completo.loc['2003-10-01':'2016-04-30'] #4592 datos
#%% plotting serie sis en 3 tramos con barplot, MALE SAL-->lo plotea como prom mensual
# Crear una figura y ejes
fig, axes = plt.subplots(figsize=(40, 14), nrows=3, ncols=1, sharey=True, facecolor='#E6E6FA')
sns.set_style('white')
# Crear un diccionario de colores para las categorías de 'periodo'
colores_periodo = {'c': 'orange', 'f': 'blue'}
#sns.set_palette(["orange", "blue"])

# Gráfico 1
ax1 = sns.barplot(x=primer_tramo_sis.index.strftime('%Y-%m'), y='Valor',
                  data=primer_tramo_sis, hue='periodo', palette= colores_periodo, ax=axes[0], 
                  saturation=0.7, ci=None)
# Agregar grid al Gráfico 1
ax1.grid(True, axis='y', linestyle='--', alpha=1)
# Gráfico 2
ax2 = sns.barplot(x=segundo_tramo_sis.index.strftime('%Y-%m'), y='Valor',
                  data=segundo_tramo_sis, hue='periodo', palette= colores_periodo, ax=axes[1], saturation=0.7, ci=None)
# Agregar grid al Gráfico 1
ax2.grid(True, axis='y', linestyle='--', alpha=1)
# Gráfico 3
ax3 = sns.barplot(x=tercer_tramo_sis.index.strftime('%Y-%m'), y='Valor',
                  data=tercer_tramo_sis, hue='periodo', palette= colores_periodo, ax=axes[2], saturation=0.7, ci=None)
# Agregar grid al Gráfico 1
ax3.grid(True, axis='y', linestyle='--', alpha=1)

for ax in [ax1, ax2, ax3]:
    ax.set_yticks(np.arange(-1.5, 2, 1))
ax3.set_xlabel("Años", fontsize=24)
# Ajusta las ubicaciones de los ticks en el eje x
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.tick_params(axis='both', labelsize=18)


ax2.set_ylabel("Índice SIS", fontsize=22)
# Agregar título solo a la figura completa
fig.suptitle('Indice SIS - Período 1979-2016', fontsize=28)

# Quitar las leyendas de los subplots
for ax in [ax1, ax2, ax3]:
    ax.get_legend().remove()
# Crear una única leyenda fuera de los subplots
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 15}, title='Periodo')
plt.tight_layout()
plt.savefig("plot/Indice_sis_completo.png", dpi=300)
plt.show()
#%% PLOTTING  DE SIS_COMPLETO en tres tramos CON FILL_BETWEEN
sns.set(font_scale=1.0, style="whitegrid")
fig, axes= plt.subplots(figsize=(20, 8), nrows=3, ncols=1, sharey=True,
                       dpi=100, facecolor='#E6E6FA')
fig.suptitle('Indice SIS - Período 1979-2016', fontsize=32)
# Crear un diccionario de colores para las categorías de 'periodo'
colores_periodo = {'c': 'violet', 'f': 'darkslateblue'}
#PRIMER TRAMO
# Inicializa variables para rastrear el inicio y final de cada segmento
inicio_segmento = 0
# Itera a través de los datos y crea segmentos para cada categoría
for i in range(1, len(primer_tramo_sis)):
    if primer_tramo_sis['periodo'].iloc[i] != primer_tramo_sis['periodo'].iloc[i - 1]:
        # Cuando cambia la categoría, crea un segmento desde el inicio_segmento hasta i-1
        axes[0].fill_between(primer_tramo_sis.index[inicio_segmento:i], 
                        primer_tramo_sis['Valor'].iloc[inicio_segmento:i], 
                        color=colores_periodo[primer_tramo_sis['periodo'].iloc[i - 1]], 
                        label=primer_tramo_sis['periodo'].iloc[i - 1])
        inicio_segmento = i
# Añade el último segmento
axes[0].fill_between(primer_tramo_sis.index[inicio_segmento:], 
                     primer_tramo_sis['Valor'].iloc[inicio_segmento:], 
                     color=colores_periodo[primer_tramo_sis['periodo'].iloc[-1]], 
                     label=primer_tramo_sis['periodo'].iloc[-1])
axes[0].grid(True, axis='y', linestyle='--', alpha=1)

#SEGUNDO TRAMO
# Inicializa variables para rastrear el inicio y final de cada segmento
inicio_segmento = 0
# Itera a través de los datos y crea segmentos para cada categoría
for i in range(1, len(segundo_tramo_sis)):
    if segundo_tramo_sis['periodo'].iloc[i] != segundo_tramo_sis['periodo'].iloc[i - 1]:
        # Cuando cambia la categoría, crea un segmento desde el inicio_segmento hasta i-1     
        axes[1].fill_between(segundo_tramo_sis.index[inicio_segmento:i], 
                        segundo_tramo_sis['Valor'].iloc[inicio_segmento:i], 
                        color=colores_periodo[segundo_tramo_sis['periodo'].iloc[i - 1]], 
                        label=segundo_tramo_sis['periodo'].iloc[i - 1])
        inicio_segmento = i
# Añade el último segmento     
axes[1].fill_between(segundo_tramo_sis.index[inicio_segmento:], 
                     segundo_tramo_sis['Valor'].iloc[inicio_segmento:], 
                     color=colores_periodo[segundo_tramo_sis['periodo'].iloc[-1]], 
                     label=segundo_tramo_sis['periodo'].iloc[-1])
axes[1].grid(True, axis='y', linestyle='--', alpha=1)     

#TERCER TRAMO   
# Inicializa variables para rastrear el inicio y final de cada segmento
inicio_segmento = 0
# Itera a través de los datos y crea segmentos para cada categoría
for i in range(1, len(tercer_tramo_sis)):
    if tercer_tramo_sis['periodo'].iloc[i] != tercer_tramo_sis['periodo'].iloc[i - 1]:
        # Cuando cambia la categoría, crea un segmento desde el inicio_segmento hasta i-1          
        axes[2].fill_between(tercer_tramo_sis.index[inicio_segmento:i], 
                        tercer_tramo_sis['Valor'].iloc[inicio_segmento:i], 
                        color=colores_periodo[tercer_tramo_sis['periodo'].iloc[i - 1]], 
                        label=tercer_tramo_sis['periodo'].iloc[i - 1])
        inicio_segmento = i
# Añade el último segmento  
axes[2].fill_between(tercer_tramo_sis.index[inicio_segmento:], 
                     tercer_tramo_sis['Valor'].iloc[inicio_segmento:], 
                     color=colores_periodo[tercer_tramo_sis['periodo'].iloc[-1]], 
                     label=tercer_tramo_sis['periodo'].iloc[-1])
axes[2].grid(True, axis='y', linestyle='--', alpha=1)
# Establece los límites de x para todos los subplots
axes[0].set_xlim('1979-10-01', '1991-10-01')
axes[1].set_xlim('1991-10-01', '2003-10-01')
axes[2].set_xlim('2003-10-01', '2016-10-01')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[0:2], labels[0:2], loc='upper right', prop={'size': 10}, title='Período')
axes[1].set_ylabel("Índice SIS", fontsize=18)
axes[2].set_xlabel("Años", fontsize=18)
# Agregar título solo a la figura completa
fig.suptitle('Indice SIS - Período 1979-2016', fontsize=24)

date_form = DateFormatter("%Y-%m")
for axes in [axes[0], axes[1], axes[2]]:
    axes.set_yticks(np.arange(-4, 5, 2))
    axes.xaxis.set_major_locator(ticker.MultipleLocator(24))
    axes.tick_params(axis='both', labelsize=18)
    axes.xaxis.set_major_locator(mdates.MonthLocator(interval=12))    
    axes.xaxis.set_major_formatter(date_form)
plt.tight_layout()
plt.savefig("plot/Indice_sis_completo_fill.png", dpi=300)
plt.show()
#%% Parte de plot anterior sobre el que me base para graficar (no correr)
# Axis labels
axes[0].set_xlabel(" ")
axes[0].tick_params(axis='both',labelsize=20)
axes[1].set_xlabel(" ")
axes[1].set_ylabel("Precipitación mayor al P75 [mm]", fontsize= 26)
axes[1].tick_params(axis='both',labelsize=20)
axes[2].set_xlabel("Años", fontsize= 28)
axes[2].tick_params(axis='both',labelsize=20)

axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=12))

# Define the date format
date_form = DateFormatter("%Y-%m")
axes[0].xaxis.set_major_formatter(date_form)
axes[1].xaxis.set_major_formatter(date_form)
axes[2].xaxis.set_major_formatter(date_form)

#plt.setp(axes[0].get_xticklabels(), rotation = 90)    
axes[0].set_yticks(range(0, 126, 25))
# Set title and labels for axes
axes[0].set(xlim=["1991", "2001"])
axes[1].set(xlim=["2001", "2011"])
axes[2].set(xlim=["2011", "2021"])
plt.tight_layout()
plt.savefig("plot/Presentación_de_datos_precip_superior_p75.png", dpi=300)
plt.show()
#%%
#%% JUNTO PRECIPITACION E INDICE SIS EN UN MISMO DF 1979-2016
''' A partir del df SIS_completo (#13352 datos) y el df lluvia_periodosis (#13362 datos) incluso
los dias con valor 0 
'''
#lluvia_periodosis tiene las fechas 29feb de los años bisiestos, analizar si ese dia llovio
#para tomar una decision respecto a ese dato
indice_fechas= lluvia_periodosis.index
valor_correspondiente=[]
for fecha in indice_fechas:
    if (fecha.month==2) & (fecha.day==29):
        valor_correspondiente.append(lluvia_periodosis.loc[fecha, 'prcp'])
        print(f"El día {fecha} se registro el valor de pp {lluvia_periodosis.loc[fecha, 'prcp']}")

#como hay fechas que tienen valores relevantes vamos a agregarlas al df SIS_completo
#que tiene 3 columnas Valor, mes y periodo
# Crear una lista para almacenar las fechas y valores correspondientes
fechas_a_agregar = []
valores_nan = []
# Iterar a través de las fechas en el índice de lluvia_periodosis
for fecha in lluvia_periodosis.index:
    if (fecha.month == 2) and (fecha.day == 29):
        fechas_a_agregar.append(fecha)
        valores_nan.append(float('nan')) #aca deberia cambiarlo pro promedio entre anterior y posterior
# Crear un nuevo DataFrame con las fechas y valores NaN
df_nuevas_fechas = pd.DataFrame({'Valor': valores_nan, 'mes': 2, 'periodo': 'c'}, 
                                index=fechas_a_agregar)  # Agregar columna 'mes' y periodo

# Concatenar el nuevo DataFrame con SIS_completo
SIS_completo = pd.concat([SIS_completo, df_nuevas_fechas], axis=0)
# Ordenar el índice si es necesario
SIS_completo.sort_index(inplace=True) #ahora SIS_completo tiene 13362 datos al igual q los datos de lluvia_periodosis
# AHORA SI PUEDO JUNTAR LOS DF Y GENERAR UNO NUEVO EN UN NUEVO ARCHIVO
SIS_completo['Fecha']=SIS_completo.index
SIS_completo['Fecha'] = pd.to_datetime(SIS_completo['Fecha'])
#voy a poner todas las horas como 12 utc
SIS_completo['Fecha']= SIS_completo['Fecha'].dt.normalize()+datetime.timedelta(hours=12)
SIS_completo.index = SIS_completo['Fecha']
#ELIMINO LA COLUMNA FECHA Y HAGO DATETIME EL INDICE FECHA COMO NADIA 
SIS_completo.drop(columns=["Fecha"], axis=1, inplace=True)
SIS_completo.index = pd.to_datetime(SIS_completo.index)
# Combinar los datos de lluvia_periodosis y SIS_completo 
datos_combinados = lluvia_periodosis.merge(SIS_completo, left_index=True, right_index=True)
#%% Guardo el nuevo DataFrame en un archivo CSV manteniendo el índice de df1
path='data_set/datos_combinados.csv'
datos_combinados.to_csv(path, index=True)
#%% Guardo el DF de intensidad mensual para periodo sis calculado antes en un archivo CSV 
path2='data_set/intensidadpp_mensual.csv'
intensidad_mensual_lluvia_periodosis.to_csv(path2, index=True)
#%% INTENTOS FALLIDOS A RESOLVER - 
'''
DEBERIA HACERLO CON BARRA Y COMPARAR 
CON LA MEDIA TRIMESTRAL SOBRE EL GRAFICO, TIPO ONDA TRIMESTRAL
puedo hacer un arange np array y usar ax.plot poniedo en x el array, 
como hice mas adelante con 'valores extremos'
'''
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import MultipleLocator
register_matplotlib_converters()
# Use white grid plot background from seaborn
sns.set(font_scale=1.8, style="whitegrid")

# Create figure and plot space
fig, ax = plt.subplots(figsize=(20, 10))

# Add x-axis and y-axis
ax.plot(acum_trimestral.index,
       acum_trimestral['prcp'],
       marker='o',
       color='m')

ax.xaxis.set_major_locator(mdates.YearLocator(byyear=(1, 3)))
ax.xaxis.set_minor_locator(mdates.YearLocator(2))
#ax.grid(True)
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Precipitación acumulada trimestral (mm)",
       title="Serie de precipitación acumulada trimestral (mm) - Ezeiza Aero", 
       xlim=["1991", "2021"])
# Define the date format
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
plt.setp(ax.get_xticklabels(), rotation = 90)  
plt.show()
   

#plt.figure(figsize=(14,6))
#lt.bar(prom_trimestral.index,prom_trimestral['prcp'], linewidth=0.2, color= 'blue')
acum_trimestral['prcp'].plot.bar(figsize=(22,6)) 
#acum_trimestral.resample().mean
plt.ylabel('Precipitacion (mm)', fontsize=14)
xfmt = mdates.DateFormatter('%m-%y')
plt.title('Lluvia trimestral (mm) - Ezeiza Aero', fontsize=16)
plt.show()

#obtengo los mismos resultados usando groupby de la siguiente manera si tuviera un solo año
#cuando tengo mas de un año, agrupo por mes independientemente del año y ploteo
f_dfclima['Fecha'].dt.quarter
mean_tri=f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).mean()
plt.bar(mean_tri.index, mean_tri['prcp'])  # ---> habria que mejorarlo y ver a que trim corresponde c/u

#%% GRAFICO EXTREMOS DE PRECIPITACION (p99) periodo climatologico  #27 extremos
valores_extremos=f_dfclima.loc[f_dfclima['prcp']>=69.7] #por encima el P99 y son 27
print(valores_extremos)
eje_x=np.array(['1992-01-01', '1992-05-07', '1993-02-09', '1993-12-04',
               '1994-12-13', '1995-04-09', '1998-01-05', '2002-04-17',
               '2003-11-11', '2005-03-08', '2005-08-23', '2007-03-02', 
               '2008-02-29', '2009-12-24', '2010-02-20', '2010-05-24',
               '2012-10-29', '2012-12-20', '2013-04-02', '2014-01-24',
               '2014-10-29', '2014-11-30', '2015-08-10', '2018-11-11', 
               '2018-12-14', '2019-07-12', '2020-01-15'])
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Use white grid plot background from seaborn
sns.set(font_scale=1.0, style="whitegrid")
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(eje_x, valores_extremos['prcp'], width= 0.4, color='mediumslateblue')
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Precipitación (mm)",
       title="Extremos de precipitación por encima del percentil 99 - Ezeiza Aero")
plt.setp(ax.get_xticklabels(), rotation = 90)  
plt.show()
#%% BOXPLOT anual de precipitación DIARIA 
#aca lo que hago es agregar una columna MES a mi f_dfclima en la cual extraigo 
#del indice que es la fecha, solo los datos del mes. 
#Luego realizar un boxplot con los meses de 1 a 12 con los valores segun la pp diaria
#por ello me quedan las cajas mas pegadas al piso (abscisa)

f_dfclima['Mes']=f_dfclima.index.month  
fig, ax = plt.subplots(figsize=(14,9), dpi=72)
sns.boxplot(data=f_dfclima,x='Mes',y='prcp', ax=ax)
plt.savefig('estacionalidad_anual.png', dpi=300)
plt.show()

f_dfclima['prcp'].quantile(0.99).round(1)
[f_dfclima['Mes']==1]['prcp'].quantile(0.5).round(1)
#%% BOXPLOT acum mensual de PRECIPITACION (version 2) COLORES PERIODO SIS
#Voy a hacer el mismo boxplot pero creando el DF mes que tiene el acumulado mensual de pp para cada año y 
#se ve mejor la variabilidad que si lo muestro con datos diarios
acum_mes=pd.DataFrame(f_dfclima.prcp.resample('M').sum())
print(acum_mes.head(10))
'''
ESTA PINTA:  
            valor
timeend          
1961-01-31  125.5
1961-02-28   85.5
1961-03-31   66.0
1961-04-30   95.0
1961-05-31  161.2
1961-06-30   33.4
1961-07-31   66.5
1961-08-31   38.8
1961-09-30   31.7
1961-10-31  114.5
'''
#vuelgo a agregar una columna con solo los meses sacada del indice, pero ahora en el df mes (antes lo hice en el f_df)
acum_mes['Mes'] = acum_mes.index.month   
fig, ax = plt.subplots(figsize=(16,8), dpi=120)
colores=['lightblue' if (4<x<10) else 'pink' for x in acum_mes.Mes]
plt.title("Ezeiza SMN - Precipitación acumulada mensual\nPeríodo 1991-2020", fontweight= 'bold', fontsize=16)
sns.boxplot(data=acum_mes, x='Mes', y='prcp', ax=ax, linewidth= 1.7, palette=colores,
            flierprops={"marker": "x"})
plt.ylabel("Precipitación (mm)", fontsize=16)
plt.xlabel("Mes", fontsize=16)
ax.tick_params(axis='both',labelsize=14)
plt.savefig('plot/boxplot_acum_mensual.png', dpi=300)
plt.show()
#%% BOXPLOT DATOS diarios de precipitacion para los eneros-periodo climatologico
f_dfclima['Año'] = f_dfclima.index.year

sns.catplot(x="Año", y="prcp", kind="box",
     data=f_dfclima[f_dfclima.Mes==1], height = 7, aspect = 3)
plt.title("Boxplot de precipitación eneros", fontweight= 'bold', fontsize=15)
plt.ylabel("Precipitación (mm)", fontsize=16)
plt.xlabel("Eneros", fontsize=16)
ax.tick_params(axis='both',labelsize=14)
plt.savefig('boxplot_pp_diaria_enero.png', dpi=300)
plt.show()
#octubres
sns.catplot(x="Año", y="prcp", kind="box",
     data=f_dfclima[f_dfclima.Mes==10], height = 7, aspect = 3)
plt.title("Boxplot de precipitación octubres", fontweight= 'bold', fontsize=15)
plt.ylabel("Precipitación (mm)", fontsize=16)
plt.xlabel("Octubres", fontsize=16)
ax.tick_params(axis='both',labelsize=14)
#junios
sns.catplot(x="Año", y="prcp", kind="box",
     data=f_dfclima[f_dfclima.Mes==6], height = 7, aspect = 3)
plt.title("Boxplot de precipitación junios", fontweight= 'bold', fontsize=15)
plt.ylabel("Precipitación (mm)", fontsize=16)
plt.xlabel("Junios", fontsize=16)
ax.tick_params(axis='both',labelsize=14)
#%%



