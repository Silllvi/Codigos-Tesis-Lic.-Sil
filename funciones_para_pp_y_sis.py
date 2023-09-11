# -*- coding: utf-8 -*-
"""
AQUÍ SOLO ENCONTRARÁS FUNCIONES CREADAS PARA DIFERENTES CALCULOS CON DATOS DE PRECIPITACIÓN E ÍNDICE sis

@author: SIL_SOSA
"""
#%%

import pandas as pd

#%%
'''#creo un indice para el rango de datos, que exceptue ciertos meses del año'''
def crear_indice_fechas(inicio, fin, meses_a_excluir):
    # Generar todas las fechas desde inicio hasta fin
    rango_fechas = pd.date_range(start=inicio, end=fin, freq='D')

    # Filtrar las fechas excluyendo los meses especificados
    indice_fechas = [fecha for fecha in rango_fechas if fecha.month not in meses_a_excluir]
  
    return indice_fechas

#%%
'''quiero crear una función que tome como argumentos las condiciones que deseo aplicar en el análisis.  
Se utilizan argumentos opcionales (`sis_condition` y `periodo_condition`). Si no se proporciona un valor para
uno de los argumentos, esa condición no se aplicará en el filtrado de datos. 
Esto me permitirá reutilizar el mismo procedimiento con diferentes conjuntos de condiciones sin tener que repetir el código,
incluyendo casos donde solo una de las condiciones es relevante.'''

def climatologia_conjunta(data, sis_condition=None, periodo_condition=None, prcp_condition=None): 
    
    '''
    Realiza un análisis de climatología de la precipitación según condiciones dadas.

    Parámetros:
    data (DataFrame): El DataFrame de datos.
    sis_condition (str): Condición para la columna 'Valor'. Si es None, no se aplica esta condición.
    periodo_condition (str): Condición para la columna 'periodo'. Si es None, no se aplica esta condición.
    precip_condition (str): Condición para la columna 'prcp'. Si es None, no se aplica esta condición.

    Retorna:
    pandas.Series: Estadísticas descriptivas de la precipitación en los datos filtrados.
    '''
    
    filtered_data = data.copy()
    
    if sis_condition is not None:
        if sis_condition == "positivos":
            filtered_data = filtered_data[filtered_data['Valor'] > 0]
        elif sis_condition == "negativos":
            filtered_data = filtered_data[filtered_data['Valor'] < 0]
        elif isinstance(sis_condition, (int, float)):
            filtered_data = filtered_data[filtered_data['Valor'] == sis_condition]
           
    if periodo_condition is not None:
        filtered_data = filtered_data[filtered_data['periodo'] == periodo_condition]
        
    if prcp_condition is not None:
        filtered_data = filtered_data[filtered_data['prcp'] > prcp_condition]

    climatologia_c = pd.DataFrame(filtered_data['prcp'].describe().round(2))
    percentiles= pd.DataFrame(filtered_data['prcp'].quantile([0.9, 0.95, 0.99]).round(2))
    climatologia_c= climatologia_c.append(percentiles)
    #quiero mover la fila de maximo hacia el final
    #la guardo en una variable temporal
    fila_temporal=climatologia_c.iloc[7]
    # Elimina la fila del DataFrame original
    climatologia_c = climatologia_c.drop(climatologia_c.index[7])
    # Inserta la fila en la nueva posición
    climatologia_c = climatologia_c.append(fila_temporal, ignore_index=True)
    nuevos_indices= ['Count', 'Mean', 'Std', 'Min', 'P25', 'P50', 'P75', 'P90', 'P95','P99', 'Max']
    climatologia_c=climatologia_c.rename(index=dict(zip(climatologia_c.index, nuevos_indices)))
    return (climatologia_c)

    
