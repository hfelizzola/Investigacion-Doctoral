import pandas as pd
import numpy as np

#---------------------------------------------------------
def year_range_ind(data,subcategoria,departamental=True):
    """
    Revisar para cada subcategoria el rango de años en los que se tienen datos disponibles
    """
    if(departamental):
        codEntidades = departamentos['Código']*1000
    else:
        codEntidades = municipios['Codigo Municipio']
        
    # Filtrar la base de datos para acceso
    dataSubcategoria = data.loc[(data['Subcategoría'] == subcategoria) & (data['Código Entidad'].isin(codEntidades))]
    
    # Verificar los años en los que se tienen registrado datos para los indicadores
    yearMinMax = dataSubcategoria.groupby('Indicador').agg({'Año':[np.min, np.max]}).reset_index()
    yearMinMax.columns = ["_".join(pair) for pair in yearMinMax.columns]
    return(yearMinMax)
#---------------------------------------------------------
def subcat_dim(data,departamental=True):
    """
    Extraer subcategorias de la dimensión
    """
    if(departamental):
        codEntidades = departamentos['Código']*1000
    else:
        codEntidades = municipios['Codigo Municipio']
    
    # Filtrar la base de datos para acceso
    dataSub = data.loc[data['Código Entidad'].isin(codEntidades)]
    
    subcatDim = list(pd.unique(dataSub['Subcategoría']))
    return(subcatDim)