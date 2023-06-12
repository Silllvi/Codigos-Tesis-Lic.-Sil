
# -*- coding: utf-8 -*-
"""

Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import datetime
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import MultipleLocator 
register_matplotlib_converters()
sbn.set() # Setting seaborn as default style even if use only matplotlib

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

primer_tramo=f_solo_lluvia.loc['1961-01-02':'1981-01-01'] 
segundo_tramo= f_solo_lluvia.loc['1981-01-02':'2001-01-01'] 
tercer_tramo= f_solo_lluvia.loc['2001-01-02':'2021-01-01'] 
cuarto_tramo= f_solo_lluvia.loc['2021-01-02':'2023-03-20'] 

#creo un df de nan para agregarlo a cuartad de datos que los otros subplots
ix=pd.date_range(start='20230321', end='20410101', freq='D')
extension= pd.DataFrame()
extension['Fecha'] =pd.to_datetime(ix,format ='%Y/%m/%d')
extension['Fecha'] = extension['Fecha'].dt.normalize()+datetime.timedelta(hours=12)
extension.index = extension['Fecha']
extension['prcp']=np.nan
extension.drop(columns=["Fecha"], axis=1, inplace=True)
extension.index = pd.to_datetime(extension.index)

cuarto_tramo=cuarto_tramo.append(extension) 


#%%
#plotting tramos prueba

plt.style.use("ggplot")

fig, axes = plt.subplots(figsize = (40,23), nrows=4, ncols=1, sharey=True, dpi= 100)
fig.suptitle('Ezeiza SMN - Serie de precipitación diaria-Enero 1961 - Marzo 2023\n   ', fontsize= 28)

#add DataFrames to subplots
axes[0].fill_between(primer_tramo.index, primer_tramo['prcp'], color='blue')
axes[1].fill_between(segundo_tramo.index, segundo_tramo['prcp'], color='blue')
axes[2].fill_between(tercer_tramo.index, tercer_tramo['prcp'], color='blue')
axes[3].fill_between(cuarto_tramo.index, cuarto_tramo['prcp'], color='blue')


# Axis labels
axes[0].set_xlabel(" ")
axes[0].tick_params(axis='both',labelsize=26)
axes[1].set_xlabel(" ")
axes[1].set_ylabel("Precipitación diaria [mm]", fontsize= 28)
axes[1].tick_params(axis='both',labelsize=26)
axes[2].set_xlabel(" ")
axes[2].tick_params(axis='both',labelsize=26)
axes[3].set_xlabel("Años", fontsize= 28)
axes[3].tick_params(axis='both',labelsize=26)

'''axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=12))'''

# Define the date format
date_form = DateFormatter("%Y-%m")
axes[0].xaxis.set_major_formatter(date_form)
axes[1].xaxis.set_major_formatter(date_form)
axes[2].xaxis.set_major_formatter(date_form)
axes[3].xaxis.set_major_formatter(date_form)

axes[0].xaxis.set_major_locator(MultipleLocator(731))
axes[0].xaxis.set_minor_locator(MultipleLocator(731))
axes[0].set(xlim=["1961-01", "1981-01"])
axes[1].xaxis.set_major_locator(MultipleLocator(731))
axes[1].xaxis.set_minor_locator(MultipleLocator(365))
axes[1].set(xlim=["1981-01", "2001-01"])
axes[2].xaxis.set_major_locator(MultipleLocator(731))
axes[2].xaxis.set_minor_locator(MultipleLocator(365))
axes[2].set(xlim=["2001-01", "2021-01"])
axes[3].xaxis.set_major_locator(MultipleLocator(731))
axes[3].xaxis.set_minor_locator(MultipleLocator(365))
axes[3].set(xlim=["2021-01", "2041-01"])
#plt.setp(axes[0].get_xticklabels(), rotation = 90)    
axes[0].set_yticks(range(0, 125, 25))

plt.tight_layout()

plt.show()
#queda acomodaa titulos, ylabel, mejorar visualizacion

#%%
'''
PERCENTILES - CONTROL DE CALIDAD - Desde aca uso la CLIMATOLOGIA

'''
#me fijo que no haya valores negativos
(df['prcp']<0).sum()

#para la CLIMATOLOGIA SELECCIONO UN INTERVALO
df_climatologia=df.loc['1991-01-02':'2021-01-01'] #30 años eL 01 DE ENERO COMO ES 12 UTC TIENE EL DATO CORRESPONDIENTE AL 31 DE DIC DE 2020
df_climatologia.info() #--->10958 de los cuales 14 NAN's y 10944 no nulos
'''
DatetimeIndex: 10958 entries, 1991-01-02 12:00:00 to 2021-01-01 12:00:00
'''
nan_rows_clima = (df_climatologia[df_climatologia.isnull().any(1)]) 
print(nan_rows_clima) #14 Nan's

#%%
'''PARA PERCENTILES VOY A CONSIDERAR SOLO LOS DIAS CON LLUVIA'''

f_dfclima = df_climatologia[df_climatologia['prcp']!=0]  #2666 rows
f_dfclima.describe()
f_dfclima_p75=f_dfclima[f_dfclima['prcp']>=15]
f_dfclima_p75=f_dfclima_p75.asfreq("D")
#calculo pm trimestral para valores mayores o iguales a P75
pm_trim_p75=f_dfclima_p75['prcp'].rolling(90, min_periods=1).mean().shift(1)

#%%
'''

#PROMEDIO MOVIL

'''
f_dfclima_pm=f_dfclima.asfreq("D")
#para los dias con lluvia tendria que dejar la frec diaria y poner esos dias como nan
#o hacerlo desde la indexacion que por lo que estoy pensando seria lo mismo.
Promedio_movil=f_dfclima_pm['prcp'].rolling(360, min_periods=1).mean().shift(1) #con shift(1) toma el promedio de los primeros 12 y lo ubica en el lugar 13
#min_periods= número mín de obs en ventana requeridas para tener un valor; de lo contrario, el resultado es
'''VER, SIGUE SIENDO RARO, GRAFICAR ENCIMA DE L SERIE Y HCERLO A MANO A VER QUE ESTA  TENDRIA Q HACER EL ACUMULADO MENSUAL???'''
#plotting
plt.figure(figsize=(22,6))
plt.plot(Promedio_movil.index,Promedio_movil, linewidth=0.8, color='green')
plt.ylabel('Precipitación diaria (mm)', fontsize=14)
plt.title('Promedio movil de 12 meses-precipitación diaria (mm) - Ezeiza Aero', fontsize=16)
plt.show()
#%%
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
Estadísticos para el periodo climatologico, difernciando trimestralmente

'''

#quiero hacer el describe diferenciando la onda anual

#promedio de pp diaria para cada trimestre
promedios_trim=f_dfclima.resample('Q').mean()
promedios_trim.plot(figsize= (15, 5), linewidth= 1.5)

#promedio trimestral de precipitacion diaria (unico)
f_dfclima['Fecha']=pd.to_datetime(f_dfclima.index, format ='%Y/%m/%d')
promedio_trimestral=f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).mean().round(1)
estadistico_trimestral=f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).agg(['mean',
                                                                             'std', 'min', 'max']).round(1)
percentiles_trimestral= f_dfclima.groupby(f_dfclima['Fecha'].dt.quarter).quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)
#%%
''' PROMEDIO MENSUAL PARA LA SERIE CLIMATOLOGICA 1991-2020'''
meses=f_dfclima.resample('M').sum() #acumulados para cada mes
meses['Fecha']=pd.to_datetime(meses.index, format ='%Y/%m/%d')
meses_prom=meses.groupby(meses['Fecha'].dt.month).mean().round(1) #promedia los acum mensuales
sbn.set(font_scale=1.0, style="whitegrid")

fig, ax = plt.subplots(figsize=(14, 5))
ax3=ax.bar(meses_prom.index, meses_prom['prcp'], width= 0.4, color='mediumslateblue')


ax.set(xlabel="Meses",
       ylabel="Precipitación acumulada (mm)",
       title="Ezeiza SMN- Valor medio de precipitación mensual\n1991-2020")
ax.set_xticks(range(1, 13, 1))

#plt.setp(ax.get_xticklabels()) 
#pone etiqueta a cada barra
for p in ax3.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., 
                                                      p.get_height()), ha='center', 
                va='center', xytext=(0, 10), textcoords='offset points')
plt.show()

#%%

#plotting ACUMULADOS MENSUALES   ----> ACOMODAR EL EJE  X

meses['mes']=meses.index.month
fig, ax=plt.subplots(figsize=(20,5))
                                               
ax.plot(meses.index, meses.prcp,
        marker='o', color='b', linestyle='--')

ax.set(xlabel="Meses",
       ylabel="Precipitación mensual (mm)",
       title="Ezeiza SMN- Precipitación Mensual\n1991-2020")


date_form = DateFormatter("%Y-%m")
ax.set_xticks(range(date_form))
#ax.axhline(y=mean_acum_trimestral[0],
 #             color='r', linestyle='--', label='media verano')

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
sbn.set(font_scale=1.0, style="whitegrid")

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
sbn.boxplot(data=f_dfclima,x='Mes',y='prcp', ax=ax)
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
               
medias_mensuales = acum_mes.groupby(acum_mes['Mes']).mean()
#?
fig, ax = plt.subplots(figsize=(14,9), dpi=120)
plt.title("Promedios mensuales de pecipitación", fontweight= 'bold', fontsize=16)
plt.bar(medias_mensuales.index, medias_mensuales['prcp'])

                
acum_mes['Mes'] = acum_mes.index.month   
fig, ax = plt.subplots(figsize=(16,8), dpi=120)
plt.title("Ezeiza SMN - Precipitación acumulada mensual\nPeríodo 1991-2020", fontweight= 'bold', fontsize=16)
sbn.boxplot(data=acum_mes, x='Mes', y='prcp', ax=ax, linewidth= 1.7, 
            flierprops={"marker": "x"})
plt.ylabel("Precipitación (mm)", fontsize=16)
plt.savefig('boxplot_acum_mensual.png', dpi=300)
ax.tick_params(axis='both',labelsize=14)
plt.show()


prom_enero=acum_mes(acum_mes['Mes']==1).mean()

prom_mensual=pd.DataFrame(clima.prcp.resample('M').mean())
prom_mensual['Mes'] = prom_mensual.index.month 
sbn.barplot(x=Mes, y=prcp, data=prom_mensual)
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



