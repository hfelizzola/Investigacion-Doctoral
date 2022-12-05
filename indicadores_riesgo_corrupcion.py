"""
Este modulo contiene funciones para calcular indicadores de riesgo de corrupción en contratación pública
"""
#%%
# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%

def integrar_indicadores(df_left, df_right, key):
    df = pd.merge(left=df_left, right=df_right, on=key, how="left")
    return(df)

def resumen_columna(df, col_grupo, col_resumen, total_contratos=False):
    """
    Genera un resumen estadístico por columna
    """
    # Función para calcular rango
    def rango(x):
        return max(x) - min(x)
    
    if total_contratos: 
        estadisticas = ['size','mean','median','min','max',np.std, rango]
    else:
        estadisticas = ['mean','median','min','max',np.std, rango]
    
    resumen = (
        df
        .groupby(col_grupo)
        .agg({col_resumen: estadisticas})
        .reset_index()
    )   



    rename_col_names = ['ind_ent_' + i[0] + '_' + i[1] if i[1] != '' else i[0] for i in resumen.columns]
    rename_col_names = ['ind_ent_total_contratos' if i.endswith('_size') else i for i in rename_col_names]
    rename_col_names = [i.lower().replace(" ","_") if i != 'Nombre de la Entidad' else i for i in rename_col_names]
    resumen.columns = rename_col_names

    return resumen

def frac_contratacion_modalidad(df, col_grupo, col_modalidad, col_cuantia, modalidad):
    """
    Calcula la fracción de contratos, en cantidad y cuantia, para diferentes modalidades de contratación
    """

    # Calcula la fracción en número de contratos
    ind_num_contratos = (
        df
        .groupby(col_grupo)
        .apply(lambda x: np.mean(x[col_modalidad] == modalidad))
        .reset_index(name='ind_ent_frac_' + modalidad + '_num')
        )

    # Calcula la fracción en cantidad de contratos
    ind_val_contratos = (
        df
        .groupby(col_grupo)
        .apply(lambda x: x[x[col_modalidad] == modalidad][col_cuantia].sum()/x[col_cuantia].sum())
        .reset_index(name='ind_ent_frac_' + modalidad + '_val')
        )
    
    #return ind_num_contratos
    return pd.merge(left=ind_num_contratos, right=ind_val_contratos, on=col_grupo)

# Función auxiliar para calcular el HHI en cada entidad entidad
def HHI_aux(df):
    total = df['total'].sum()
    hhi_elements = [((value/total)*100)**2 for value in df['total']]
    hhi = sum(hhi_elements)
    return hhi


def HHI_index(df, col_grupo, col_id_contratista, col_cuantia):
    """
    Calcula el Índice  Herfindahl-Hirschman HHI por grupo: entidad, mercado, otros...
    """
    # Calcula el HHI en numero de contratos
    ind_hhi_num = (
        df
        .groupby([col_grupo, col_id_contratista])
        .size()
        .reset_index(name='total')
        .groupby(col_grupo)
        .apply(HHI_aux)
        .reset_index()
        .rename({0:"ind_ent_HHI_num"}, axis=1)
        )

    # Calcula el HHI en cuantia de contratos
    ind_hhi_val = (
        df
        .groupby([col_grupo,col_id_contratista])[col_cuantia]
        .sum()
        .reset_index(name='total')
        .groupby(col_grupo)
        .apply(HHI_aux)
        .reset_index()
        .rename({0:"ind_ent_HHI_val"}, axis=1)
        )

    return pd.merge(left=ind_hhi_num, right=ind_hhi_val, on=col_grupo, how='left')


def diversidad_simpson_aux(df):
    """
    Función auxiliar para calcular el indice de diversidad de Simpson
    """
    total = df['total'].sum()
    ID_elements = [value*(value-1) for value in df['total']]
    ID = sum(ID_elements)/(total*(total-1))
    return ID


def simpson_index(df, col_grupo, col_id_contratista):
    """
    Calcula el índice de diversidad de simpson según numero de contratos de cada contratista
    """
    return (
        df
        .groupby([col_grupo,col_id_contratista])
        .size()
        .reset_index(name='total')
        .groupby(col_grupo)
        .apply(diversidad_simpson_aux)
        .reset_index(name='ind_ent_div_simpson_num')
    )

def compute_IC4K(df):
    """
    Función auxiliar para calcular la suma de las participaciones de los 4 principales contratistas
    """
    total = df['total'].sum()
    top_4 = df['total'].sort_values(ascending = False)[0:4]    
    ICK4 = sum(top_4)/total
    return ICK4

def participacion_cuatro_principales(df, col_grupo, col_id_contratista, col_cuantia):
    """
    Calcula la partipación de los cuatro principales contratistas de acuerdo al  
    número de contratos y el valor total de los contratos
    """

    ind_IC4k_num = (
    df
    .groupby([col_grupo,col_id_contratista])
    .size()
    .reset_index(name='total')
    .groupby(col_grupo)
    .apply(compute_IC4K)
    .reset_index(name='ind_ent_IC4K_num')
    )


    # Calcula el ICK4 en cuantia de contratos
    ind_IC4k_val = (
    df
    .groupby([col_grupo,col_id_contratista])[col_cuantia]
    .sum()
    .reset_index(name='total')
    .groupby(col_grupo)
    .apply(compute_IC4K)
    .reset_index(name='ind_ent_IC4K_val')
    )

    return pd.merge(left=ind_IC4k_num, right=ind_IC4k_val, on=col_grupo, how='left')

def frac_contratos_adiciones(df, col_grupo, col_adiciones_tiempo, col_adiciones_cuantia):
    """
    Calcula la fracción de contratos con adiciones por grupo
    """
    ind_frac_adiciones_tiempo = (
        df[[col_grupo, col_adiciones_tiempo]]
        .assign(adicion_tiempo = lambda x: np.int64(x[col_adiciones_tiempo] > 0))
        .groupby(col_grupo)['adicion_tiempo']
        .mean()
        .reset_index(name='ind_ent_frac_adicion_tiempo')
    )

    ind_frac_adiciones_cuantia = (
        df[[col_grupo,col_adiciones_cuantia]]
        .assign(adicion_cuantia = lambda x: np.int64(x[col_adiciones_cuantia] > 0))
        .groupby(col_grupo)['adicion_cuantia']
        .mean()
        .reset_index(name='ind_ent_frac_adicion_cuantia')
    )

    return pd.merge(left=ind_frac_adiciones_tiempo, right=ind_frac_adiciones_cuantia, on=col_grupo, how='left')

def razon_ncontratistas_ncontratos(df, col_grupo, col_id_contratista):
    ind_razon_ncontratistas_ncontratos = (
        df
        .groupby([col_grupo,col_id_contratista])
        .size()
        .reset_index(name='total')
        .groupby(col_grupo)
        .apply(lambda x: x[col_id_contratista].nunique()/x['total'].sum())
        .reset_index(name='ind_ent_razon_contratistas_ncontratos')
    )

    return ind_razon_ncontratistas_ncontratos
