"""
Este modulo contiene una serie de funciones que realiza procesos de limpieza, filtrado y estandarizacion de datos del df I
"""

# imports
# %%
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
# %%

def filtrar_estado_proceso(df, col_name, categorias):
    """
    Filtra el df según el estado del proceso
    """

    # Estado inicial
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Inicial')
    print('Número de contratos: {:,}'.format(contratos_inicial))
    print(df[col_name].value_counts())

    # Filtro
    df = df.loc[df[col_name].isin(categorias)]

    # Estado Final
    contratos_final = df.shape[0]
    diferencia_contratos = contratos_inicial - contratos_final

    print('-'*50)
    print('Resumen Final')
    print('Número de contratos:  {:,}'.format(contratos_final))
    print('Contratos retirados: {:,}'.format(diferencia_contratos))
    print(df[col_name].value_counts())
    print('-'*50)

    return df


def filtrar_valor_contrato(df, col_name, min=0, max=np.infty):
    """
    Filtra los contratos de df I de acuerdo a un valor mínimo y un valor máximo
    """

    df['rango_cuantia'] = pd.cut(x=df[col_name],
                                 bins=[0, 1e6, 1e7, 1e8, 1e9,
                                       1e10, 1e11, np.infty],
                                 labels=['Menos de 1M', '[1M - 10M)', '[10M - 100M)', '[100M - 1,000M)',
                                         '[1,000M - 10,000M', '[10,000M - 100,000M)', 'Mas de 100,000M'],
                                 right=False)

    # Estado inicial
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Inicial')
    print('Número de contratos: {:,}'.format(contratos_inicial))
    print(df['rango_cuantia'].value_counts(sort=False))

    # Filtro
    df = df[(df[col_name] >= min) & (df[col_name] <= max)]

    # Estado Final
    contratos_final = df.shape[0]
    diferencia_contratos = contratos_inicial - contratos_final

    print('-'*50)
    print('Resumen Final')
    print('Número de contratos:  {:,}'.format(contratos_final))
    print('Contratos retirados: {:,}'.format(diferencia_contratos))
    print(df['rango_cuantia'].value_counts(sort=False))
    print('-'*50)

    return df


def filtrar_entidades_min_ncontratos(df, col_name, min_contratos=5):
    """
    Filtra los contratos de entidades que tienen un mínimo volumen de contratación
    """

    # Estado inicial
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Inicial')
    print('Número de contratos: {:,}'.format(contratos_inicial))
    print('Entidades: {}'.format(df[col_name].nunique()))

    # Filtro
    temp = df.groupby(col_name).size().reset_index().rename(
        {0: 'n_contratos'}, axis=1)
    temp = temp.loc[temp['n_contratos'] >= min_contratos]
    df = df.loc[df[col_name].isin(temp[col_name])]

    # Estado final
    contratos_final = df.shape[0]
    diferencia_contratos = contratos_inicial - contratos_final
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Final')
    print('Número de contratos: {:,}'.format(contratos_final))
    print('Contratos retirados: {:,}'.format(diferencia_contratos))
    print('Entidades: {}'.format(df[col_name].nunique()))

    return df


def limpiar_tipo_proceso(df, col_name):
    """
    Estandariza las categorias para el tipo de proceso o modalidad de contratación
    """

    # Estado inicial
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Inicial')
    print(df[col_name].value_counts())

    # Limpieza
    map_tipo_proceso = {
        "Concurso de Méritos Abierto": "concurso_meritos",
        "Concurso de Méritos con Lista Corta": "concurso_meritos",
        "Concurso de Méritos con Lista Multiusos": "concurso_meritos",
        "Contratación Directa (Ley 1150 de 2007)": "contratacion_directa",
        "Contratación Directa (con ofertas)": "contratacion_directa",
        "Contratación directa": "contratacion_directa",
        "Contratación Mínima Cuantía": "minima_cuantia",
        "Mínima cuantía": "minima_cuantia",
        "Licitación Pública": "licitacion_publica",
        "Licitación obra pública": "licitacion_publica",
        "Licitación pública": "licitacion_publica",
        "Régimen Especial": "regimen_especial",
        "Contratación régimen especial": "regimen_especial",
        "Contratación régimen especial (con ofertas)": "regimen_especial",
        "Selección Abreviada de Menor Cuantía (Ley 1150 de 2007)": "seleccion_abreviada",
        "Selección Abreviada del literal h del numeral 2 del artículo 2 de la Ley 1150 de 2007": "seleccion_abreviada",
        "Selección Abreviada servicios de Salud": "seleccion_abreviada",
        "Seleccion Abreviada Menor Cuantia Sin Manifestacion Interes": "seleccion_abreviada",
        "Selección Abreviada de Menor Cuantía": "seleccion_abreviada",
        "Selección abreviada subasta inversa": "seleccion_abreviada",
        "Subasta": "subasta",
        "Contratos y convenios con más de dos partes": "otros",
        "Asociación Público Privada": "otros"
    }

    df[col_name].replace(to_replace=map_tipo_proceso, inplace=True)

    # Estado Final
    print('-'*50)
    print('Resumen Final')
    print(df[col_name].value_counts())
    print('-'*50)

    return df


def limpiar_nivel_entidad(df, col_name):
    """
    Estandariza las categorias para el nivel de la entidad
    """

    # Estado inicial
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Inicial')
    print(df[col_name].value_counts())

    # Limpieza
    map_nivel_entidad = {
        "NACIONAL": "nacional",
        "TERRITORIAL": "territorial",
        "No Definida": "no_definida"
    }

    df[col_name].replace(to_replace=map_nivel_entidad, inplace=True)

    # Estado Final

    print('-'*50)
    print('Resumen Final')
    print(df[col_name].value_counts())
    print('-'*50)

    return df


def limpiar_orden_entidad(df, col_name):
    """
    Estandariza las categorias para el orden de la entidad
    """

    # Estado inicial
    print('-'*50)
    contratos_inicial = df.shape[0]
    print('Resumen Inicial')
    print(df[col_name].value_counts())

    # Limpieza
    map_orden_entidad = {
        "DISTRITO CAPITAL": "distr_capital",
        "NACIONAL CENTRALIZADO": "nac_centralizado",
        "NACIONAL DESCENTRALIZADO": "nac_descentralizado",
        "TERRITORIAL DEPARTAMENTAL CENTRALIZADO": "terri_dep_centr",
        "TERRITORIAL DEPARTAMENTAL DESCENTRALIZADO": "terri_dep_no_centr",
        "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 1": "terri_distr_1",
        "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 2": "terri_distr_2",
        "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 3": "terri_distr_3",
        "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 4": "terri_distr_4",
        "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 5": "terri_distr_5",
        "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 6": "terri_distr_6",
        "No Definido": "no_defindo"}

    df[col_name].replace(to_replace=map_orden_entidad, inplace=True)

    # Estado Final

    print('-'*50)
    print('Resumen Final')
    print(df[col_name].value_counts())
    print('-'*50)

    return df


def asignar_orden(df, col_name_entidad, col_name_orden):
    # Verificar si la entidad tiene registrado mas de un nivel
    print("Hay {} entidades con mas de un orden de entidad registrado".format(
        sum(df.groupby(col_name_entidad)[col_name_orden].nunique() > 1)))

    temp = df.groupby(col_name_entidad)[col_name_orden].nunique()
    temp = temp[temp > 1]
    temp2 = df.loc[df[col_name_entidad].isin(temp.index)].groupby(
        [col_name_entidad, col_name_orden]).size().reset_index().rename({0: "n"}, axis=1)

    if(temp2.shape[0] > 0):
        def sel_orden(groups):
            n_orden = groups.sort_values(by="n", ascending=False)
            orden = n_orden[col_name_orden][0:1]
            return orden

        temp3 = temp2.groupby(col_name_entidad).apply(sel_orden).reset_index()

        for i in range(temp3.shape[0]):
            df.loc[df[col_name_entidad] == temp3[col_name_entidad]
                   [i], [col_name_orden]] = temp3[col_name_orden][i]

        # Verificar si la entidad tiene registrado mas de un nivel
        print("Despues de la limpieza hay {} entidades con mas de un orden de entidad registrado".format(
            sum(df.groupby(col_name_entidad)[col_name_orden].nunique() > 1)))

    return df


def limpiar_dep_entidad(df, col_name):
    df[col_name] = df[col_name].str.lower().str.replace(pat="\\-|\\.", repl="_")
    df[col_name] = df[col_name].str.replace("bogotá_d_c_", "bogotá")
    print('-'*50)
    print(df[col_name].value_counts())
    
    return df

def limpiar_regimen_contratacion(df, col_name):
    
    # Estado inicial
    print('-'*50)
    print('Resumen Inicial')
    print(df[col_name].value_counts())

    map_regimen = {
        "Estatuto General de Contratación":"general",
        "Ley 80 de 1993": "general",
        "Régimen Especial": "especial"
    }

    df[col_name].replace(to_replace=map_regimen, inplace=True)

     # Estado Final

    print('-'*50)
    print('Resumen Final')
    print(df[col_name].value_counts())
    print('-'*50)

    return df


def limpiar_razon_social_contratista(df, col_name):
    
    print('Antes de procesar hay {:,} contratistas unicos según su razón social'.format(df[col_name].nunique()))

    # Convertir a minuscula
    df['razon_social_contratista_mod'] = df[col_name].str.lower()

    # Eliminar espacios innecesarios
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.strip()
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.split()
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.join(' ')

    # Quitar caracteres especiales
    car_especiales = '\\.|\\:|\\;|\\,|\\-|\\_|\\@|\\*|\\+|\\^|\\¿|\\?|\\¡|\\!|\\$|\\%|\\/|\\||\\#|\\(|\\)|\\[|\\]|\\{|\\}|\\=|\\º|\\ª|\\´|\\`'
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.replace(pat=car_especiales,repl="")

    # Reemplazar tildes
    vocales_tilde = ['á|à','é|è','í|ì','ó|ò','ú|ù']
    vocales =['a','e','i','o','u']
    for voc, voc_til in zip(vocales, vocales_tilde):
        df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.replace(pat=voc_til, repl=voc)

    # Cambiar & por y
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.replace(pat='\\&',repl='y')

    # Cambiar las terminaciones del tipo de sociedad
    pat_sociedades = ' sa$| s a$| s  a$| sas$| s a s$| sa s$| s as$| sac$| s a c$| s c$| sc$| ltda$| ltoa$| limitada$| limtada$| s$'
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.replace(pat=pat_sociedades,repl='')

    # Vocales o números al final
    df['razon_social_contratista_mod'] = df['razon_social_contratista_mod'].str.replace(pat=' [aeiou]$| [0-9]$',repl='')

    print('Antes de procesar hay {:,} contratistas unicos según su razón social'.format(df['razon_social_contratista_mod'].nunique()))

    return df

def limpiar_id_contratista(df, col_name_id, col_name_razon_social):
    # Quitar espacios en los extremos
    df['id_contratista_mod'] = df[col_name_id].str.strip()
    df['id_contratista_mod'] = df['id_contratista_mod'].str.split()
    df['id_contratista_mod'] = df['id_contratista_mod'].str.join(' ')
    df['id_contratista_mod'] = df['id_contratista_mod'].str.lower()

    # Quitar caracteres especiales
    car_especiales = '\\.|\\:|\\;|\\,|\\-|\\_|\\@|\\*|\\+|\\^|\\¿|\\?|\\¡|\\!|\\$|\\%|\\/|\\||\\#|\\(|\\)|\\[|\\]|\\{|\\}|\\=|\\º|\\ª|\\´|\\`|\\&'
    df['id_contratista_mod'] = df['id_contratista_mod'].str.replace(pat=car_especiales,repl="")

    # Reemplazar tildes
    vocales_tilde = ['á|à','é|è','í|ì','ó|ò','ú|ù']
    vocales =['a','e','i','o','u']
    for voc, voc_til in zip(vocales, vocales_tilde):
        df['id_contratista_mod'] = df['id_contratista_mod'].str.replace(pat=voc_til, repl=voc)

    # Intercambiar id y razon social
    #razon_social_contratista_mod = ['12 34','mary aparicio','tico felizzola', '5678','tico felizzola','mary aparicio']
    #id_contratista_mod = ['mary aparicio','1234','5678','tico felizzola','5678','1234']
    #df = pd.DataFrame({col_name_razon_social: razon_social_contratista_mod, 'id_contratista_mod':id_contratista_mod})
    #print(df)

    df['is_id_contratista_mod_alpha'] = df['id_contratista_mod'].str.split()
    df['is_id_contratista_mod_alpha'] = df['is_id_contratista_mod_alpha'].str.join('')
    df['is_id_contratista_mod_alpha'] = df['is_id_contratista_mod_alpha'].str.isalpha()
    df['id_contratista_mod_2'] = df['id_contratista_mod']

    df['is_razon_social_contratista_mod_numeric'] = df[col_name_razon_social].str.split()
    df['is_razon_social_contratista_mod_numeric'] = df['is_razon_social_contratista_mod_numeric'].str.join('')
    df['is_razon_social_contratista_mod_numeric'] = df['is_razon_social_contratista_mod_numeric'].str.isnumeric()
    df['razon_social_contratista_mod_2'] = df[col_name_razon_social]


    df.loc[(df['is_razon_social_contratista_mod_numeric']) & (df['is_id_contratista_mod_alpha']), [col_name_razon_social]] = df.loc[(df['is_razon_social_contratista_mod_numeric']) & (df['is_id_contratista_mod_alpha']), ['id_contratista_mod_2']]
    df.loc[(df['is_razon_social_contratista_mod_numeric']) & (df['is_id_contratista_mod_alpha']), ['id_contratista_mod']] = df.loc[(df['is_razon_social_contratista_mod_numeric']) & (df['is_id_contratista_mod_alpha']), ['razon_social_contratista_mod_2']]
    df.drop(columns=['id_contratista_mod_2','razon_social_contratista_mod_2','is_id_contratista_mod_alpha','is_razon_social_contratista_mod_numeric'], inplace=True)

    # Extraer solo numeros
    df['id_contratista_mod'] = df['id_contratista_mod'].str.extract(r'(\d+)')

    # Cortar los ID a 10 digitos máximo
    df['id_contratista_mod'] = list(map(lambda x: str(x), df['id_contratista_mod']))
    df['id_contratista_mod'] = list(map(lambda x: x[0:10], df['id_contratista_mod']))

    print('Contratistas unicos por id: {:,}'.format(df[col_name_id].nunique()))
    print('Contratistas unicos por id modificado: {:,}'.format(df['id_contratista_mod'].nunique()))

    return df

def convertir_duracion_contrato_dias(df, col_duracion, col_tipo_duracion):
    # Transformar el plazo de ejecución del contrato de meses a días
    df['duracion_dias'] = np.where(df[col_tipo_duracion] ==  "M", df[col_duracion]*30, df[col_duracion])

    return df




def unificar_adiciones_tiempo(df, col_add_dias, col_add_meses):
    df["adicion_tiempo_dias_mod"] = df[col_add_dias] + df[col_add_meses]*30

    return df
