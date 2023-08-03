
# -*- coding: utf-8 -*-
"""

Spyder Editor

This is a temporary script file.
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
#%%
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
#%%
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
#%%
'''copio y extiendo datos a actualidad y me quedo con un solo df que va a ser la suma de los
2 y ahi ya puedo aplicar prom movil y hacer climatologia'''
#%%
'''
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
#%%
#plotting SERIE COMPLETA

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
#%%
#probanding plotting
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
#%%
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
#%%
'''Para visulizarla mejor se va a partir la serie total en 3'''

primer_tramo=f_solo_lluvia.loc['1961-01-02':'1981-09-27'] #7574 datos
segundo_tramo= f_solo_lluvia.loc['1981-09-28':'2002-06-23'] #7574 datos
tercer_tramo= f_solo_lluvia.loc['2002-06-24':'2023-03-20'] #7575 datos

#%%
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

#%%
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

#%%
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
#%%
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
#%%
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

#%%
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
#%%
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
#%%
''' PROMEDIO MENSUAL PARA LA SERIE CLIMATOLOGICA 1991-2020'''
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


#clrs = ['grey' if (4< x < 10) else 'red' for x in meses_prom.index ]
#prom=sbn.barplot(x=meses_prom.index, y=meses_prom['prcp'], palette=clrs)
#prom.set(xlabel ="GFG X", ylabel = "GFG Y", title ='some title')


#fig, ax = plt.subplots(figsize=(14, 5))

#ax3=ax.bar(meses_prom.index, meses_prom['prcp'], width= 0.4, color='mediumslateblue')
#ax3=ax.bar(lista_meses_calidos, meses_prom['prcp'], width= 0.4, color='pink')

'''
ax.set(xlabel="Meses",
       ylabel="Precipitación acumulada (mm)",
       title="Ezeiza SMN- Valor medio de precipitación mensual\n1991-2020")
ax.set_xticks(range(1, 13, 1))
'''
#plt.setp(ax.get_xticklabels()) 
#pone etiqueta a cada barra
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig("plot/promedio_mensual_pp.png", dpi=200)
plt.show()

#%%
'''TABLA DE ESTADÍSTICOS SEGUN SIS VERANO O SIS INVIERNO'''
'''período climatologico 1991-2020'''

f_dfclima['Fecha']= pd.to_datetime(f_dfclima['Fecha']) #estan solo los dias de lluvia

#separo los datos en meses fríos y meses cálidos
meses_sis_frio=f_dfclima[f_dfclima['Fecha'].dt.month.isin([5, 6, 7, 8, 9])] #991 datos
meses_sis_calido=f_dfclima[f_dfclima['Fecha'].dt.month.isin([1, 2, 3, 4, 10, 11, 12])] #1675

estadisticos_sis_frio=meses_sis_frio.agg(['mean', 'std', 'min', 'max']).round(1)
estadisticos_sis_calido=meses_sis_calido.agg(['mean', 'std', 'min', 'max']).round(1)

percentiles_sis_frio=meses_sis_frio.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)
percentiles_sis_calido=meses_sis_calido.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)

data_clima_sis={'epoca_sis':['SIS CÁLIDO', 'SIS FRÍO'], 'media':[12.9, 9.1],
                'Std': [16.5, 13], 'P75': [17, 12], 'P90': [34.9, 26], 'P95': [48, 34.9],
                'P99': [72.7, 55.6],
                'Max': [116, 120.3]}

climatologia_epoca_sis=pd.DataFrame(data_clima_sis)
climatologia_epoca_sis= climatologia_epoca_sis.set_index('epoca_sis')
climatologia_epoca_sis=climatologia_epoca_sis.transpose()
climatologia_epoca_sis.to_excel("plot/tabla_climatologiapp_epoca_sis.xlsx")

plt.style.use("ggplot")
columnas=climatologia_epoca_sis.columns
num_barras= len(columnas)
ancho_barras=0.8
indice=climatologia_epoca_sis.index
colores=['#C79FEF', '#7BC8F6']
climatologia_epoca_sis.plot(kind='bar', figsize=(10, 4), color=colores)

#configurar l titulo y las etiquetas de los ejes
plt.title('Ezeiza SMN - Climatología segun época SIS - 1991-2020', fontsize=12)
plt.xlabel('Época SIS', fontsize=14)
plt.ylabel('Precipitación diaria (mm)', fontsize=11)
#Mostrar la leyenda
plt.legend(columnas, fontsize=9)
plt.savefig('plot/Climatologia_epoca_sis.png', dpi=200)
plt.show()
#%%

#PERCENTIL 75 PARA CADA MES
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
#%%
#plotting tramos prueba
filtro_datos_p75['mes']=filtro_datos_p75['Fecha'].dt.month

filtro_1=filtro_datos_p75.loc['1991-01-21':'2001-01-01'] #3634 dttos
filtro_2=filtro_datos_p75.loc['2001-01-02':'2010-12-14'] # 3634 datos
filtro_3=filtro_datos_p75.loc['2010-12-15':'2020-11-25'] #3634 datos

plt.style.use("ggplot")

fig, axes = plt.subplots(figsize = (36,18), nrows=3, ncols=1, sharey=True, dpi= 100, facecolor='silver')
fig.suptitle('Ezeiza SMN - Extremos de precipitación\n1991 - 2020', fontsize= 33)

clrs1 = ['lightblue' if (4< x < 10) else 'pink' for x in filtro_1.mes]
clrs2 = ['lightblue' if (4< x < 10) else 'pink' for x in filtro_2.mes]
clrs3 = ['lightblue' if (4< x < 10) else 'pink' for x in filtro_3.mes]
#add DataFrames to subplots
axes[0].plot(filtro_1.index, filtro_1['prcp'], 'o', color='red')
axes[0].fill_between(filtro_1.index, filtro_1['prcp'], color='maroon', linewidth=1.5)
axes[1].plot(filtro_2.index, filtro_2['prcp'], 'o', color='red')
axes[1].fill_between(filtro_2.index, filtro_2['prcp'], color='maroon', linewidth=1.5)
axes[2].plot(filtro_3.index, filtro_3['prcp'], 'o', color='red')
axes[2].fill_between(filtro_3.index, filtro_3['prcp'], color='maroon', linewidth=1.5)

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
plt.savefig("plot/Presentación_de_datos_precip_superior_p75.png", dpi=300)
plt.show()


#%%
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import MultipleLocator
register_matplotlib_converters()


# Create figure and plot space
fig, ax = plt.subplots(figsize=(50, 12), facecolor='lavender')
fig.suptitle('Ezeiza SMN - Extremos de precipitación\n1991 - 2020', fontsize= 28)
# Add x-axis and y-axis
filtro_datos_p75['mes']=filtro_datos_p75.index.month  
clrs2 = ['blue' if (4< x < 10) else 'red' for x in filtro_datos_p75.mes]  #ETA MAL LO DEL COLOR

fig, ax = plt.subplots(figsize=(50, 12))
ax.plot(filtro_datos_p75.index, filtro_datos_p75['prcp'], color="black")
colors = ['pink', 'blue']
mes=np.array(list(filtro_datos_p75.index.month))
for i in range(mes):
    plt.fill_between(filtro_datos_p75['Fecha'].iloc[i:i+4], filtro_datos_p75['prcp'].iloc[i:i+4], color=colors[i])

plt.grid(True)
ax.fill_between(filtro_datos_p75.index, filtro_datos_p75['prcp'], 
                color= 'red')
ax.fill_between(filtro_datos_p75.index, filtro_datos_p75['prcp'], 
                where=filtro_datos_p75.mes > 10, facecolor= 'pink')
ax.fill_between(filtro_datos_p75.index, filtro_datos_p75['prcp'], 
                where=filtro_datos_p75.mes <4, facecolor= 'pink')

ax.xaxis.set_major_locator(MultipleLocator(731))
ax.xaxis.set_minor_locator(MultipleLocator(365))

# Set title and labels for axes
ax.set(xlim=["1991-01", "2020-12"])

ax.tick_params(axis='both',labelsize=22)
ax.set_ylabel("Precipitación diaria por encima de percentil 75 [mm]", fontsize= 26)
ax.tick_params(axis='both',labelsize=24)
ax.set_xlabel("Años", fontsize= 26)
ax.tick_params(axis='both',labelsize=24)
ax.set_yticks(range(10, 130, 10))

plt.setp(ax.get_xticklabels(), rotation = 0)     

# Define the date format
date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.show()

'''
plt.style.use("ggplot")
fig, ax=plt.subplots(figsize=(20,5))
ax=filtro_datos_p75['prcp'].plot(kind='bar', color='blue')

plt.setp(ax.get_xticklabels(), rotation = 90)     

# Define the date format


ax.set(xlabel="Años",
       ylabel="Precipitación diaria (mm)",
       title="Ezeiza SMN- Precipitación extrema\n1991-2020")
locator=ticker.MultipleLocator(base=24)
plt.gca().xaxis.set_major_locator(locator)
'''
#%% 
'''promedio de pp diaria para cada MES INTENSIDAD'''
#RESAMPLE.MEAN CALCULA LA MEDIA DE LOS GRUPOS, EXCLUYENDO VALOR FALTANTE
#---> MI GRUPO ES UN MES, ENTONCES AL ACUM DE 30 DIAS LO DIVIDE POR 
# LA CANT DE DIAS QUE LLOVIO -->INTENSIDAD (mm/dia)
intensidad_mensual=filtro_datos_p75.resample('M').mean().round(1)
mes_int=intensidad_mensual['mes']
colors= ['grey' if (4< s < 10) else 'red' for s in mes_int]
fig, ax = plt.subplots(figsize=(50,12))
sns.set_style('white')
ax=sns.barplot(x=intensidad_mensual.index, y='prcp',
               data=intensidad_mensual, palette=colors)

plt.title('Ezeiza SMN - Intensidad de precipitación mensual', fontsize= 28)
#plotting INTENSIDAD
ax = plt.gca()
ax.legend(labels=["SIS FRÍO","SIS CÁLIDO"], fontsize='20')
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

ax.tick_params(axis='both',labelsize=22)
ax.set_xlabel("Años", fontsize= 26)
ax.set_ylabel("Intensidad (mm/día)", fontsize= 26)
ax.grid(True)

plt.setp(ax.get_xticklabels(), rotation =45)  
plt.savefig("plot/Intensidad_mensual.png", dpi=300)
plt.show()

#%%

#Percentil 75 para cada mes diferenciando por criterio sis
################## FRIO ######################

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


#nuevo=meses_sis_calido[(meses_sis_calido.index.month==10) & (meses_sis_calido.prcp>=15.9)]
#%%
#plotting ACUMULADOS MENSUALES   ----> ACOMODAR EL EJE  X
#podri servir para comparar con el promedio mensual de pp acumulada para 
#cada mes, el grafico rosa y celeste de barras. Lo dejamos por las duda
# nos sirva en algun momento
acumulado_mensual['mes']=acumulado_mensual.index.month

#meses['mes']=meses.index.month
fig, ax=plt.subplots(figsize=(20,5))
                                               
ax.plot(acumulado_mensual.index, acumulado_mensual.prcp,
        marker='o', color='b', linestyle='--')

ax.set(xlabel="Meses",
       ylabel="Precipitación mensual (mm)",
       title="Ezeiza SMN- Precipitación Mensual\n1991-2020")


date_form = DateFormatter("%Y-%m")
ax.set_xticks(range(date_form))
#ax.axhline(y=mean_acum_trimestral[0],
 #             color='r', linestyle='--', label='media verano')
#%%
#PP TRIMESTRAL ACUMULADA VS EL PROMEDIO DE PP TRIMESTRAL PARA CADA ESTACION
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

#%% 
'''LEO Y ABRO ARCHIVOS CORRESPONDIENTES A VARIABLE SIS OA
Y LOS ACOMODO QUEDANDOME UN DATAFRAME CON INDICE FECHA Y COLUMNA DE VALORES SIS
'''
directorio = 'data_set'
archivo2_sis_OA= 'SIS7914_ONDEFMA1090.csv'
fname2 = os.path.join(directorio,archivo2_sis_OA)
dfSIS_OA= pd.read_csv(fname2, sep= ';', encoding='latin-1') #/216, 1025)

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

#plotting
fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_OA, palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 1979-2016 considerando solo meses de Octubre a Abril',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)

fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_OA.loc['2012':'2016'], palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 2012-2016 considerando solo meses de Octubre a Abril',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)
#%%
'''
AHORA CALCULO CLIMATOLOGIA SEPARANDO EN VALORES POSITIVOS Y VALORES NEGATIVOS DEL SIS_OA
'''
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
describe_val_pos_OA=valores_positivos_sis_OA.describe().round(1)
percentiles_valores_positivos_sis_OA=valores_positivos_sis_OA.quantile([0.9, 0.95, 0.99]).round(1)
describe_val_pos_OA=describe_val_pos_OA.append(percentiles_valores_positivos_sis_OA)
describe_val_pos_OA=describe_val_pos_OA.rename(columns={'Valor':'SIS-OA-Positivo'})
describe_val_pos_OA.to_excel('plot/descripcion_valores_pos_OA.xlsx', index=True)


# FILTRAR VALORES NEGATIVOS
valores_negativos_sis_OA = dfSIS_OA[dfSIS_OA['Valor'] < 0]['Valor']
valores_negativos_sis_OA =pd.DataFrame(valores_negativos_sis_OA)
print("Valores negativos en la columna {}:\n{}".format('Valor', valores_negativos_sis_OA)) #3896 datos negativos
describe_val_neg_OA=valores_negativos_sis_OA.describe().round(1)
percentiles_valores_negativos_sis_OA=valores_negativos_sis_OA.quantile([0.9, 0.95, 0.99]).round(1)
describe_val_neg_OA=describe_val_neg_OA.append(percentiles_valores_negativos_sis_OA)
describe_val_neg_OA=describe_val_neg_OA.rename(columns={'Valor':'SIS-OA-Negativo'})
describe_val_neg_OA.to_excel('plot/descripcion_valores_neg_OA.xlsx', index=True)

#junto las 2 tablas de dataframe de estadisticos. positivas y negativas
combined_describe_sisOA = pd.concat([describe_val_pos_OA, describe_val_neg_OA], axis=1)
combined_describe_sisOA.to_excel('plot/tabla_estadisticos_sisOA_negypos.xlsx', index=True)
#aca deberia graficar como con sis calido y frio
combined_describe_sisOA=combined_describe_sisOA.drop('count', axis=0)
#quiero mover la fila de maximo hacia el final
#la guardo en una variable temporal
fila_temporal=combined_describe_sisOA.iloc[6]
# Elimina la fila del DataFrame original
combined_describe_sisOA = combined_describe_sisOA.drop(combined_describe_sisOA.index[6])
# Inserta la fila en la nueva posición
combined_describe_sisOA = combined_describe_sisOA.append(fila_temporal, ignore_index=True)
nuevos_indices= ['Mean', 'Std', 'Min', 'P25', 'P50', 'P75', 'P90', 'P95','P99', 'Max']
combined_describe_sisOA=combined_describe_sisOA.rename(index=dict(zip(combined_describe_sisOA.index, nuevos_indices)))
#%%

indice=combined_describe_sisOA.index
sns.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Estadísticos para los meses de Octubre a Abril del índice SIS\nPeríodo 1979-2016', fontsize=12)
ax = sns.barplot(data=combined_describe_sisOA, x=indice, y=combined_describe_sisOA["SIS-OA-Negativo"], 
            color='dodgerblue', label="SIS-OA-Negativo", ax=ax)
sns.barplot(data=combined_describe_sisOA, x=indice, y=combined_describe_sisOA["SIS-OA-Positivo"],
                 color='#C79FEF', label="SIS-OA-Positivo")

ax.legend(loc='lower right', fontsize=9)
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
                va=va,  xytext=xytext, textcoords='offset points', fontsize=10, color= color, fontweight='normal')

plt.savefig('plot/Estadisticos_indice_sisOA.png', dpi=200)
plt.show()
#%%
'''Ahora similar pero con boxplot'''

# Crea una figura y dos ejes, uno para valores positivos y otro para valores negativos
fig, (ax_positivos, ax_negativos) = plt.subplots(1, 2, sharey=True, figsize=(14,6))
fig.suptitle('Valores positivos y negativos del índice SIS\n Octubre a Abril, Período 1979-2016', fontsize=14)
# Grafica los boxplots para los valores positivos y negativos
ax_positivos.boxplot(valores_positivos_sis_OA , vert=False, flierprops={'marker':'x'}, 
                     showfliers= False, showmeans=True, meanline= True, notch=True, patch_artist=True,
                     boxprops=dict(facecolor='#C79FEF', color='black'), 
                     meanprops= dict(linestyle='-', linewidth=1.5, color='green'))
ax_negativos.boxplot(valores_negativos_sis_OA , vert=False, flierprops={'marker':'x'}, 
                     showfliers= False, showmeans=True, meanline= True, notch=True, patch_artist=True,
                     boxprops=dict(facecolor='dodgerblue', color='black'),
                     meanprops= dict(linestyle='-', linewidth=1.5, color='green'))


# Establece el título y etiquetas del gráfico
ax_positivos.set_title('Valores Positivos', fontsize=12)
ax_positivos.set_xlabel('Índice SIS')
ax_positivos.set_ylabel(' ')
ax_positivos.set_yticklabels([' ', ' '])


ax_negativos.set_title('Valores Negativos', fontsize=12)
ax_negativos.set_xlabel('Índice SIS')
ax_negativos.set_ylabel(' ')

 
# Muestra el gráfico
plt.tight_layout() #asegura que este todo bien ajustado
plt.savefig('plot/boxplot_indice_SIS.png', dpi=300)
plt.show()
#%%
archivo3_sis_MS= 'SIS8015_MJJAS1090.csv'
fname3 = os.path.join(directorio,archivo3_sis_MS)
dfSIS_MS= pd.read_csv(fname3, sep= ';', encoding='latin-1') #(156, 37)

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

#plotting
fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_OA, palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 1979-2016 considerando solo meses de Octubre a Abril',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)

fig = plt.subplots(figsize=(20, 5))
sns.lineplot(data=dfSIS_OA.loc['2012':'2016'], palette=['magenta'], linewidth=0.9, linestyle='dotted').set(title='Índice SIS para el periodo 2012-2016 considerando solo meses de Octubre a Abril',
        xlabel='Años', ylabel='Valor del índice')
sns.set_theme(style='white', font_scale=1)
#%%

'''
  
INTENTOS FALLIDOS A RESOLVER - DEBERIA HACERLO CON BARRA Y COMPARAR 
CON LA MEDIA TRIMESTRAL SOBRE EL GRAFICO, TIPO ONDA TRIMESTRAL
puedo hacer un arange np array y usar ax.plot poniedo en x el array, 
como hice mas adelante con 'valores extremos'
'''
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import MultipleLocator
register_matplotlib_converters()
# Use white grid plot background from seaborn
sbn.set(font_scale=1.8, style="whitegrid")

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
f_df['fecha'].dt.quarter
mean_tri=f_df.groupby(f_df['fecha'].dt.quarter).mean()
plt.bar(mean_tri.index, mean_tri['prcp'])  # ---> habria que mejorarlo y ver a que trim corresponde c/u

#%%
'''
GRAFICO EXTREMOS DE PP
'''
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

# Define the date format
#date_form = DateFormatter("%Y-%m-%d")
#ax.xaxis.set_major_formatter(date_form)
plt.setp(ax.get_xticklabels(), rotation = 90)  

plt.show()
#%%
'''
BOXPLOT mensual
'''
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
#%%
'''
BOXPLOT acum mensual (version 2)
'''

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
#medias_mensuales = acum_mes.groupby(acum_mes['Mes']).mean()
#?
#fig, ax = plt.subplots(figsize=(14,9), dpi=120)
#plt.title("Promedios mensuales de pecipitación", fontweight= 'bold', fontsize=16)
#plt.bar(medias_mensuales.index, medias_mensuales['prcp'])

   
fig, ax = plt.subplots(figsize=(16,8), dpi=120)
colores=['lightblue' if (4<x<10) else 'pink' for x in acum_mes.Mes]
plt.title("Ezeiza SMN - Precipitación acumulada mensual\nPeríodo 1991-2020", fontweight= 'bold', fontsize=16)
sns.boxplot(data=acum_mes, x='Mes', y='prcp', ax=ax, linewidth= 1.7, palette=colores,
            flierprops={"marker": "x"})
plt.ylabel("Precipitación (mm)", fontsize=16)
plt.savefig('plot/boxplot_acum_mensual.png', dpi=300)
ax.tick_params(axis='both',labelsize=14)
plt.show()


prom_enero=acum_mes(acum_mes['Mes']==1).mean()

prom_mensual=pd.DataFrame(clima.prcp.resample('M').mean())
prom_mensual['Mes'] = prom_mensual.index.month 
sns.barplot(x=Mes, y=prcp, data=prom_mensual)
#%%
#cuando tengo mas de un año, agrupo por mes independientemente del año y ploteo
clima['fecha'].dt.month
mean_month=clima.groupby(clima['fecha'].dt.month).mean()
plt.bar(mean_month.index, mean_month['prcp'])





#%%



#acum_diario['día']= acum_diario.index.day
#acum_diario[acum_diario['Mes']==1].describe()

fig, ax = plt.subplots(figsize=(14,9), dpi=120)
plt.title("Boxplot de precipitación acumulada diaria", fontweight= 'bold', fontsize=16)
sns.boxplot(data=acum_diario[acum_diario.Mes==1], x="día", y='prcp', ax=ax)
plt.ylabel("Precipitación acum. diaria para Enero (mm)", fontsize=14)
plt.savefig('boxplot_acum_diario_enero.png', dpi=300)
plt.show()



sns.catplot(x="Año", y="prcp", kind="box",
     data=clima[clima.Mes==1], height = 3, aspect = 8)
plt.savefig('boxplot_diario_enero.png', dpi=300)
plt.title("Boxplot de precipitación eneros", fontweight= 'bold', fontsize=15)
plt.show()

clima[clima.Mes==1].describe()   #### ESTA ES LA UE VA!!
fig, ax = plt.subplots(figsize=(14,9), dpi=120)
plt.title("Boxplot de precipitación acumulada mensual", fontweight= 'bold', fontsize=16)
sbn.boxplot(data=clima[clima.Mes==1], x='Año', y='prcp', ax=ax)
                       plt.ylabel("Precipitación acum. mensual (mm)", fontsize=14)
plt.savefig('boxplot_acum_mensual.png', dpi=300)
plt.show()


from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(prom_mensual, model='additive')
fig = decomposition.plot()
plt.show()
#%%

'''
BOXPLOT DATOS eneros''' #vuelvo a f_df
f_df['Año'] = f_df.index.year 

sns.catplot(x="Año", y="valor", kind="box",
     data=f_df[f_df.Mes==1], height = 3, aspect = 8)
plt.title("Boxplot de precipitación eneros", fontweight= 'bold', fontsize=15)
plt.savefig('boxplot_acum_diario_enero.png', dpi=300)
plt.show()


#%%



