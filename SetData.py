#%%
import pandas as pd
import numpy as np

#%%
def load_data():
    """Cargar datos de entrenamiento con información de indicadores de riesgos de corrupción,
    indicadores de terridata, entre otros.
    """
    
    # 1. Cargar datos
    path = "Datos/bases_integradas.csv"
    data = pd.read_csv(path)
    
    # 2. Eliminar algunas columnas y renombrar otras
    del_columns = ['nombre_entidad','nit_entidad','municipio_entidad','dep_codigo_divipola','municipio_codigo_divipola']
    data.drop(columns=del_columns, inplace=True)
    
    # 3. Renombrar algunas columnas
    data.rename(columns={'departamento_entidad':'dep', 'orden_entidad': 'orden'}, inplace=True)
    
    # 4. Combinar departamentos con conteo que representen menos del 1% de los hospitales
    min_percent_contracts = 0.1
    depCont = data.groupby('dep').size().sort_values()/data.shape[0]
    combinar = list(depCont.index[depCont < min_percent_contracts])
    data.loc[data.dep.isin(combinar),'dep'] = 'Otro'
    
    print(data.info())
    
    return(data)

#%%

def preprocess(data, y_pred):
    """
    Realiza el preprocesamiento de las variables y define las variables X y Y
    """
       
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    
    varsPred = ['perc_adic_valor','perc_adic_tiempo']
    varsCat = list(data.dtypes.loc[data.dtypes == 'object'].index)
    varsNum = list(data.dtypes.loc[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index)
    varsNum = [x for x in varsNum if x not in varsPred]

    # 1. Extraer las variables a predecir
    Y = data[varsPred]
    
    # 2. Extraer las variables numericas
    X_num = data[varsNum]
    
    # 3. Imputar valores faltantes con la mediana
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_num)
    X_num = imp.transform(X_num)
    
    # 4. Estandarizar las variables numericas
    estandarizar = StandardScaler().fit(X_num)
    X_num = estandarizar.transform(X_num)
    X_num = pd.DataFrame(X_num, columns=varsNum)
    
    # 5. Crear variables dummies
    X_cat = pd.get_dummies(data[varsCat])
    
    # 6. Integrar variables numericas y dummies
    X = pd.concat([X_cat, X_num], axis=1)

    # Dividir los conjuntos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, Y[y_pred], test_size=0.20, random_state=777)

    # 7. Borrar los subconjuntos creados
    X_cat = None 
    X_num = None
    
    # 8. Resultado
    return(X_train, X_test, y_train, y_test)
