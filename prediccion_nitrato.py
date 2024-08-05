# Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid', font_scale=1.4)
import warnings
warnings.filterwarnings('ignore')
import time
import itertools
import re

#NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

################################### TRATAMIENTO DE LA TABLA DE CP ###################################

# Tabla de codigos postales, codigo municipio, codigo provincia
dfmunicipiosCP = pd.read_table('/datos_CP/FicheroRelacion_INE_CodigoPostal.txt', encoding='latin-1', delimiter=';')

# Tabla codigo y literal de provincia
dfprovincias = pd.read_excel('/datos_CP/codprov.xls', header=1)

# Union tablas para completar la tabla con CPs, codigos y literales municipios y provincias
dfmunicipiosCP_completo = dfmunicipiosCP.merge(dfprovincias, left_on='CodProvincia', right_on='CODIGO', how='left')

# Eliminacion columna codigo duplicada y renombrar el campo LITERAL a Provincia
dfmunicipiosCP_completo.drop(columns=['CODIGO'], inplace=True)
dfmunicipiosCP_completo.rename(columns={'LITERAL': 'Provincia'}, inplace=True)

# Guardado de la tabla
dfmunicipiosCP_completo.to_csv('/datos_CP/dfmunicipiosCP.csv', sep=';', na_rep='N/A', index=False)

################################### TRATAMIENTO DE LA TABLA DEL SINAC ###################################

# Tabla del SINAC del año 2016
df2016 = pd.read_csv('/datos_SINAC/datos_red_2016.csv', sep=';', encoding='utf-8')

# Comprobacion todos los registros corresponden a 2016
df2016['ANYO']=2016

# Estandarizar el campo CCAA para las tablas de todos los años
df2016['COMUNIDAD_AUTONOMA_UBICACION'] = df2016['COMUNIDAD_AUTONOMA_UBICACION'].str[3:]

# Renombrar todos los campos para que concuerden todos los años
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2016.columns = new_names

# Transformacion fecha y obtencion mes
df2016['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2016['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2016['MES'] = df2016['FECHA_TOMA_MUESTRA'].dt.month

# Transformacion decimales
df2016['VALOR_CUANTIFICADO'] = df2016['VALOR_CUANTIFICADO'].str.replace(',', '.')
df2016['VALOR_CUANTIFICADO'] = pd.to_numeric(df2016['VALOR_CUANTIFICADO'], errors='coerce').round(2)

# Filtrar por el parametro Nitrato
df2016_nitrato = df2016[df2016['PARAMETRO']=='Nitrato']

# Tabla del SINAC del año 2017
df2017 = pd.read_csv('/datos_SINAC/datos_red_2017.csv', sep=';', encoding='utf-8')

# Transformaciones para estandarizar todas las tablas de todos los años
df2017['ANYO']=2017
df2017['Comunidad Autónoma'] = df2017['Comunidad Autónoma'].str[3:]
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2017.columns = new_names
df2017['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2017['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2017['MES'] = df2017['FECHA_TOMA_MUESTRA'].dt.month
df2017['VALOR_CUANTIFICADO'] = df2017['VALOR_CUANTIFICADO'].str.replace(',', '.')
df2017['VALOR_CUANTIFICADO'] = pd.to_numeric(df2017['VALOR_CUANTIFICADO'], errors='coerce').round(2)
df2017_nitrato = df2017[df2017['PARAMETRO']=='Nitrato']

# Tabla del SINAC del año 2018
df2018 = pd.read_csv('/datos_SINAC/datos_red_2018.csv', sep=';', encoding='utf-8')

# Transformaciones para estandarizar todas las tablas de todos los años
df2018['ANYO']=2018
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2018.columns = new_names
# Función para convertir mayúsculas a minúsculas excepto la primera letra
def capitalize_except_first(s):
    return s[0] + s[1:].lower()
# Aplicar la función a la columna 'CCAA'
df2018['CCAA'] = df2018['CCAA'].apply(capitalize_except_first)
df2018['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2018['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2018['MES'] = df2018['FECHA_TOMA_MUESTRA'].dt.month
df2018['VALOR_CUANTIFICADO'] = df2018['VALOR_CUANTIFICADO'].str.replace(',', '.')
df2018['VALOR_CUANTIFICADO'] = pd.to_numeric(df2018['VALOR_CUANTIFICADO'], errors='coerce').round(2)
df2018_nitrato = df2018[df2018['PARAMETRO']=='Nitrato']

# Tabla del SINAC del año 2019
df2019 = pd.read_csv('/datos_SINAC/datos_red_2019.csv', sep=';', encoding='utf-8')

# Transformaciones para estandarizar todas las tablas de todos los años
df2019['ANYO']=2019
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2019.columns = new_names
df2019['CCAA'] = df2019['CCAA'].str[3:]
df2019['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2019['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2019['MES'] = df2019['FECHA_TOMA_MUESTRA'].dt.month
df2019['VALOR_CUANTIFICADO'] = df2019['VALOR_CUANTIFICADO'].str.replace(',', '.')
df2019['VALOR_CUANTIFICADO'] = pd.to_numeric(df2019['VALOR_CUANTIFICADO'], errors='coerce').round(2)
df2019_nitrato = df2019[df2019['PARAMETRO']=='Nitrato']

# Tabla del SINAC del año 2020
df2020 = pd.read_csv('/datos_SINAC/datos_red_2020.csv', sep=';', encoding='utf-8')

# Transformaciones para estandarizar todas las tablas de todos los años
df2020['ANYO']=2020
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2020.columns = new_names
df2020['CCAA'] = df2020['CCAA'].str[3:]
df2020['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2020['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2020['MES'] = df2020['FECHA_TOMA_MUESTRA'].dt.month
df2020['VALOR_CUANTIFICADO'] = df2020['VALOR_CUANTIFICADO'].round(2)
df2020_nitrato = df2020[df2020['PARAMETRO']=='Nitrato']

# Tabla del SINAC del año 2021
df2021 = pd.read_csv('/datos_SINAC/datos_red_2021.csv', sep=';', encoding='utf-8')

# Transformaciones para estandarizar todas las tablas de todos los años
df2021['ANYO']=2021
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2021.columns = new_names
df2021['CCAA'] = df2021['CCAA'].str[3:]
df2021['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2021['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2021['MES'] = df2021['FECHA_TOMA_MUESTRA'].dt.month
df2021['VALOR_CUANTIFICADO'] = df2021['VALOR_CUANTIFICADO'].str.replace(',', '.')
df2021['VALOR_CUANTIFICADO'] = pd.to_numeric(df2021['VALOR_CUANTIFICADO'], errors='coerce').round(2)
df2021_nitrato = df2021[df2021['PARAMETRO']=='Nitrato']

# Tabla del SINAC del año 2022
df2022 = pd.read_csv('/datos_SINAC/datos_red_2022.csv', sep=';', encoding='utf-8')

# Transformaciones para estandarizar todas las tablas de todos los años
df2022['ANYO']=2022
new_names=['CCAA', 'PROVINCIA', 'MUNICIPIO', 'RED', 'OPERADOR_RED', 'PUNTO_MUESTREO', 'FECHA_TOMA_MUESTRA', 'LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS',
       'PARAMETRO', 'VALOR_CUANTIFICADO', 'VALOR_RD', 'UNIDAD', 'ANYO']
df2022.columns = new_names
df2022['CCAA'] = df2022['CCAA'].str[3:]
df2022['MUNICIPIO'] = df2022['MUNICIPIO'].str.upper()
df2022['FECHA_TOMA_MUESTRA'] = pd.to_datetime(df2022['FECHA_TOMA_MUESTRA'], format='%d/%m/%Y %H:%M:%S')
df2022['MES'] = df2022['FECHA_TOMA_MUESTRA'].dt.month
df2022['VALOR_CUANTIFICADO'] = df2022['VALOR_CUANTIFICADO'].str.replace(',', '.')
df2022['VALOR_CUANTIFICADO'] = pd.to_numeric(df2022['VALOR_CUANTIFICADO'], errors='coerce').round(2)
df2022_nitrato = df2022[df2022['PARAMETRO']=='Nitrato']

# Tabla SINAC completa: Union de las tablas de todos los años en una
df_sinac = pd.concat([df2016_nitrato, df2017_nitrato, df2018_nitrato, df2019_nitrato, df2020_nitrato, df2021_nitrato, df2022_nitrato], ignore_index=True)

# Función para convertir mayúsculas a minúsculas excepto la primera letra
def capitalize_except_first(s):
    return s[0] + s[1:].lower()
# Aplicar la función a la columna 'MUNICIPIO'
df_sinac['MUNICIPIO'] = df_sinac['MUNICIPIO'].apply(capitalize_except_first)
df_sinac = df_sinac.drop(columns=['VALOR_RD', 'UNIDAD'])

# Diccionario de reemplazos
replacements = {
    'Alava': 'Araba/Álava',
	'Albacete': 'Albacete',
	'Alicante/Alacant': 'Alicante/Alacant',
	'Almería': 'Almería',
	'Araba/Álava': 'Araba/Álava',
	'Asturias': 'Asturias',
	'Avila': 'Ávila',
	'Ávila': 'Ávila',
	'Badajoz': 'Badajoz',
	'Balears (Illes)': 'Balears, Illes',
	'Balears, Illes': 'Balears, Illes',
	'Barcelona': 'Barcelona',
	'Bizkaia': 'Bizkaia',
	'Burgos': 'Burgos',
	'Cáceres': 'Cáceres',
	'Cádiz': 'Cádiz',
	'Cantabria': 'Cantabria',
	'Castellón/Castelló': 'Castellón/Castelló',
	'Ceuta': 'Ceuta',
	'Ciudad Real': 'Ciudad Real',
	'Córdoba': 'Córdoba',
	'Coruña (A)': 'Coruña, A',
	'Coruña, A': 'Coruña, A',
	'Cuenca': 'Cuenca',
	'Gipuzkoa': 'Gipuzkoa',
	'Girona': 'Girona',
	'Granada': 'Granada',
	'Guadalajara': 'Guadalajara',
	'Guipúzcoa': 'Gipuzkoa',
	'Huelva': 'Huelva',
	'Huesca': 'Huesca',
	'Jaén': 'Jaén',
	'León': 'León',
	'Lleida': 'Lleida',
	'Lugo': 'Lugo',
	'Madrid': 'Madrid',
	'Málaga': 'Málaga',
	'Melilla': 'Melilla',
	'Murcia': 'Murcia',
	'Navarra': 'Navarra',
	'Ourense': 'Ourense',
	'Palencia': 'Palencia',
	'Palmas (Las)': 'Palmas, Las',
	'Palmas, Las': 'Palmas, Las',
	'Pontevedra': 'Pontevedra',
	'Rioja (La)': 'Rioja, La',
	'Rioja, La': 'Rioja, La',
	'Salamanca': 'Salamanca',
	'Santa Cruz de Tenerife': 'Santa Cruz de Tenerife',
	'Segovia': 'Segovia',
	'Sevilla': 'Sevilla',
	'Soria': 'Soria',
	'Tarragona': 'Tarragona',
	'Teruel': 'Teruel',
	'Toledo': 'Toledo',
	'Valencia/València': 'Valencia/València',
	'Valladolid': 'Valladolid',
	'Vizcaya': 'Bizkaia',
	'Zamora': 'Zamora',
	'Zaragoza': 'Zaragoza'
}

# Reemplazar los valores en 'PROVINCIA' usando el diccionario de reemplazos
df_sinac['PROVINCIA'] = df_sinac['PROVINCIA'].map(replacements)

# Tabla de CPs, provincias y municipios
dfmunicipiosCP = pd.read_csv('/datos_CP/dfmunicipiosCP.csv', encoding='utf-8', delimiter=';')

# Vectorización TF-IDF
vectorizer = TfidfVectorizer().fit(dfmunicipiosCP['Municipio'])
tfidf_matrix = vectorizer.transform(dfmunicipiosCP['Municipio'])

# KNN para encontrar la mejor coincidencia
nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(tfidf_matrix)

# Función para encontrar la mejor coincidencia usando TF-IDF y KNN
def find_best_match_tfidf(row, column_left):
    vector = vectorizer.transform([row[column_left]])
    dist, idx = nn.kneighbors(vector, return_distance=True)
    return dfmunicipiosCP.iloc[idx[0][0]]['Municipio']

# Aplicar la función a cada fila de la tabla del SINAC
df_sinac['coincidente'] = df_sinac.apply(find_best_match_tfidf, axis=1, column_left='MUNICIPIO')

# Realizar el left join utilizando las mejores coincidencias entre la tabla del SINAC y la tabla de CPs
df_sinac_CP = df_sinac.merge(dfmunicipiosCP, left_on=['coincidente', 'PROVINCIA'], right_on=['Municipio', 'Provincia'], how='left')

# Eliminar la columna de coincidencias temporales
df_sinac_CP = df_sinac_CP.drop(columns=['coincidente'])

# Guardado en local de la tabla final
df_sinac_CP.to_csv('/datos_ETL/df_sinac_CP.csv', sep=';', na_rep='N/A', index=False)

################################### TRATAMIENTO DE LA TABLA DE COMPLEJOS PRTR ###################################

# Tabla de complejos industriales de PRTR 
dfcomplejos = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Complejos.xml', encoding='utf-8')

# Diccionario de reemplazos
replacements = {
    'Alava': 'Araba/Álava',
	'Albacete': 'Albacete',
	'Alicante/Alacant': 'Alicante/Alacant',
	'Almería': 'Almería',
	'Araba/Álava': 'Araba/Álava',
	'Asturias': 'Asturias',
	'Avila': 'Ávila',
	'Ávila': 'Ávila',
	'Badajoz': 'Badajoz',
	'Balears (Illes)': 'Balears, Illes',
	'Balears, Illes': 'Balears, Illes',
	'Barcelona': 'Barcelona',
	'Bizkaia': 'Bizkaia',
	'Burgos': 'Burgos',
	'Cáceres': 'Cáceres',
	'Cádiz': 'Cádiz',
	'Cantabria': 'Cantabria',
	'Castellón/Castelló': 'Castellón/Castelló',
	'Ceuta': 'Ceuta',
	'Ciudad Real': 'Ciudad Real',
	'Córdoba': 'Córdoba',
	'Coruña (A)': 'Coruña, A',
	'Coruña, A': 'Coruña, A',
	'Cuenca': 'Cuenca',
	'Gipuzkoa': 'Gipuzkoa',
	'Girona': 'Girona',
	'Granada': 'Granada',
	'Guadalajara': 'Guadalajara',
	'Guipúzcoa': 'Gipuzkoa',
	'Huelva': 'Huelva',
	'Huesca': 'Huesca',
	'Jaén': 'Jaén',
	'León': 'León',
	'Lleida': 'Lleida',
	'Lugo': 'Lugo',
	'Madrid': 'Madrid',
	'Málaga': 'Málaga',
	'Melilla': 'Melilla',
	'Murcia': 'Murcia',
	'Navarra': 'Navarra',
	'Ourense': 'Ourense',
	'Palencia': 'Palencia',
	'Palmas (Las)': 'Palmas, Las',
	'Palmas, Las': 'Palmas, Las',
	'Pontevedra': 'Pontevedra',
	'Rioja (La)': 'Rioja, La',
	'Rioja, La': 'Rioja, La',
	'Salamanca': 'Salamanca',
	'Santa Cruz de Tenerife': 'Santa Cruz de Tenerife',
	'Segovia': 'Segovia',
	'Sevilla': 'Sevilla',
	'Soria': 'Soria',
	'Tarragona': 'Tarragona',
	'Teruel': 'Teruel',
	'Toledo': 'Toledo',
	'Valencia/València': 'Valencia/València',
	'Valladolid': 'Valladolid',
	'Vizcaya': 'Bizkaia',
	'Zamora': 'Zamora',
	'Zaragoza': 'Zaragoza'
}

# Reemplazar los valores en 'Provincia' usando el diccionario de reemplazos
dfcomplejos['Provincia'] = dfcomplejos['Provincia'].map(replacements)

# Tabla de CPs, provincias y municipios
dfmunicipiosCP = pd.read_csv('/datos_CP/dfmunicipiosCP.csv', encoding='utf-8', delimiter=';')

# Realizar el left join de la tabla de complejos PRTR con la tabla de CPs
dfcomplejos_comp = dfcomplejos.merge(dfmunicipiosCP, left_on=['CodPostal', 'Provincia'], right_on=['CodigoPostal', 'Provincia'], how='left')

# Actualizar 'Municipio_x' combinando con 'Municipio_y'
dfcomplejos_comp['Municipio_x'] = dfcomplejos_comp['Municipio_x'].combine_first(dfcomplejos_comp['Municipio_y'])

# Eliminar 'Municipio_y' ya que no es necesaria
dfcomplejos_comp.drop(columns=['Municipio_y'], inplace=True)

# Renombrar el nombre de 'Municipio_x' por 'Municipio'
dfcomplejos_comp.rename(columns={'Municipio_x': 'Municipio'}, inplace=True)

# Eliminar la columna repetida 'CodigoPostal'
dfcomplejos_comp.drop(columns=['CodigoPostal'], inplace=True)

# Guardado en local de la tabla final
dfcomplejos_comp.to_csv('/datos_ETL/dfcomplejos_comp.csv', sep=';', na_rep='N/A', index=False)

################################### TRATAMIENTO DE LA TABLA DE EMISIONES PRTR ###################################

# Tabla emisiones PRTR del año 2016
dfemisiones2016 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2016.xml', encoding='utf-8')

# Tabla emisiones PRTR del año 2017
dfemisiones2017 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2017.xml', encoding='utf-8')

# Tabla emisiones PRTR del año 2018
dfemisiones2018 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2018.xml', encoding='utf-8')

# Tabla emisiones PRTR del año 2019
dfemisiones2019 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2019.xml', encoding='utf-8')

# Tabla emisiones PRTR del año 2020
dfemisiones2020 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2020.xml', encoding='utf-8')

# Tabla emisiones PRTR del año 2021
dfemisiones2021 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2021.xml', encoding='utf-8')

# Tabla emisiones PRTR del año 2022
dfemisiones2022 = pd.read_xml('/datos_PRTR/PRTR_Espana_MITECO_Emisiones2022.xml', encoding='utf-8')

# Tabla emisiones PRTR completa: Union de las tablas de todos los años
df_emisiones = pd.concat([dfemisiones2016, dfemisiones2017, dfemisiones2018, dfemisiones2019, dfemisiones2020, dfemisiones2021, dfemisiones2022], ignore_index=True)

# Eliminacion de la columna 'Metodo_MCE' ya que no aporta informacion relevante
df_emisiones.drop(columns=['Metodo_MCE'],inplace=True)

# Tabla complejos PRTR
dfcomplejos_comp = pd.read_csv('/datos_ETL/dfcomplejos_comp.csv', sep=';', encoding='utf-8')

# Union de la tabla de emisiones PRTR con la tabla de complejos PRTR
df_emisiones_CP = df_emisiones.merge(dfcomplejos_comp, left_on='CodigoPRTR', right_on='CodigoPRTR', how='left')

# Eliminar columna duplicada y renombrarla
df_emisiones_CP.drop(columns=['NombreDelComplejo_y'], inplace=True)
df_emisiones_CP.rename(columns={'NombreDelComplejo_x': 'NombreDelComplejo'}, inplace=True)

# Guardado en local de la tabla final de emisiones y complejos PRTR
df_emisiones_CP.to_csv('/datos_ETL/df_emisiones_CP.csv', sep=';', na_rep='N/A', index=False)

################################### TRATAMIENTO DE LA TABLA PARA MODELADO ###################################

# Tabla del SINAC final
df_sinac_CP = pd.read_csv('/datos_ETL/df_sinac_CP.csv', sep=';', encoding='utf-8')

# Tabla de emisiones y complejos PRTR
df_emisiones_CP = pd.read_csv('/datos_ETL/df_emisiones_CP.csv', sep=';', encoding='utf-8')

# Union mediante left join de la tabla del SINAC con la tabla del PRTR
df_nitrato = df_sinac_CP.merge(df_emisiones_CP, left_on=['ANYO', 'CodigoPostal', 'PROVINCIA', 'CodProvincia', 'CodMunicipio'], right_on=['AnyoReferencia', 'CodPostal', 'Provincia', 'CodProvincia', 'CodMunicipio'], how='left')

# Eliminar filas donde 'NombreDelComplejo' es NaN
df_nitrato[df_nitrato['NombreDelComplejo'].isna()]
df_nitrato = df_nitrato.dropna(subset=['NombreDelComplejo'])

# Eliminar columnas innecesarias y renombrar columnas
df_nitrato.drop(columns=['LABORATORIO', 'CLASE_BOLETIN', 'TIPO_ANALISIS', 'CodigoPRTR', 'CodPRTRDEI', 'CodIPPC', 'Direccion', 'Poblacion'], inplace=True)
df_nitrato.rename(columns={'CCAA_x': 'CCAA', 'CodigoPostal': 'CP_x', 'CodPostal': 'CP_y','Provincia': 'Provincia_y'}, inplace=True)

# Guardado de la tabla completa 
df_nitrato.to_csv('/datos_ETL/df_nitrato_completo.csv', sep=';', na_rep='N/A', index=False)

# Lectura de la tabla
df_nitrato_unique = pd.read_csv('/datos_ETL/df_nitrato_completo.csv', sep=';', encoding='utf-8')

# Comprobar si todas las filas son iguales entre 'CP_x' y 'CP_y'
all_equal = (df_nitrato_unique['CP_x'] == df_nitrato_unique['CP_y']).all()

# Mostrar el resultado de la comparación
print("¿Todas las filas son iguales?:", all_equal)

df_nitrato_unique[df_nitrato_unique['CP_x'] != df_nitrato_unique['CP_y']]

# Comprobar si todas las filas son iguales entre 'MUNICIPIO' y 'Municipio_y'
all_equal = (df_nitrato_unique['MUNICIPIO'] == df_nitrato_unique['Municipio_y']).all()

# Mostrar el resultado de la comparación
print("¿Todas las filas son iguales?:", all_equal)

df_nitrato_unique[df_nitrato_unique['MUNICIPIO'] != df_nitrato_unique['Municipio_y']]

# Comprobar si todas las filas son iguales entre 'ANYO' y 'AnyoReferencia'
all_equal = (df_nitrato_unique['ANYO'] == df_nitrato_unique['AnyoReferencia']).all()

# Mostrar el resultado de la comparación
print("¿Todas las filas son iguales?:", all_equal)

# Selecciona de columnas para la tabla destinada al modelado
columnas_seleccion = ['VALOR_CUANTIFICADO', 'ANYO', 'MES',
                      'Contaminante', 'CantidadTotalkgporaño', 'MedioReceptor', 'CNAE-2009']
df_nitrato_pred = df_nitrato_unique[columnas_seleccion]
df_nitrato_pred.rename(columns={'VALOR_CUANTIFICADO': 'ValorNitrato', 'ANYO': 'Anyo', 'MES': 'Mes'}, inplace=True)

# Eliminar duplicados
df_nitrato_pred = df_nitrato_pred.drop_duplicates()

# Guardado de la tabla para el modelado
df_nitrato_pred.to_csv('/tfm_vscode/df_nitrato_prediccion.csv', sep=';', na_rep='N/A', index=False)

################################### MODELADO ###################################

# Tabla de modelado de la concentracion de nitrato
df_nitrato = pd.read_csv('/tfm_vscode/df_nitrato_prediccion.csv', sep=';', encoding='utf-8')

# Renombrar las columnas para quitar caracteres dificiles de procesar
df_nitrato = df_nitrato.rename(columns = lambda x:re.sub('[ñ_]+', 'ny', x))
df_nitrato = df_nitrato.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Establecer una semilla para la reproducibilidad
np.random.seed(12345)

# Función para muestrear el 10% de cada grupo
def sample_group(group):
    return group.sample(frac=0.1)

# Aplicar el muestreo a cada grupo ('Contaminante', 'MedioReceptor')
nitrato_sample = df_nitrato.groupby(['Contaminante', 'MedioReceptor'], group_keys=False).apply(sample_group).reset_index(drop=True)
nitrato_sample.to_csv('/tfm_vscode/nitrato_sample.csv', sep=';', na_rep='N/A', index=False)

# Eliminar aquellos valores de 'MedioReceptor' que supongan menos del 5% de valores en la muestra
nitrato_sample['MedioReceptor'].value_counts()
nitrato_sample = nitrato_sample[nitrato_sample['MedioReceptor']!='Suelo']
nitrato_sample['MedioReceptor'].value_counts()

# Crear una tabla pivote
pivoted_df = nitrato_sample.pivot_table(index=[col for col in nitrato_sample.columns if col not in ['Contaminante', 'CantidadTotalkgporanyo']],
                                    columns='Contaminante',
                                    values='CantidadTotalkgporanyo',
                                    aggfunc='mean').reset_index()

# Aplanar las columnas en caso de que tengan múltiples niveles
pivoted_df.columns = [col if not isinstance(col, tuple) else col[1] for col in pivoted_df.columns]

# Renombrar las columnas para quitar caracteres dificiles de procesar
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[á_]+', 'a', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[é_]+', 'e', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[í_]+', 'i', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[ó_]+', 'o', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[ú_]+', 'u', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[Á_]+', 'A', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[É_]+', 'E', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[Í_]+', 'I', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[Ó_]+', 'O', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[Ú_]+', 'U', x))
pivoted_df = pivoted_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Rellenar los valores NaN con ceros de la tabla pivotada
df_nitrato_pred = pivoted_df.fillna(0)
df_nitrato_pred.to_csv('/tfm_vscode/df_nitrato_muestra.csv', sep=';', na_rep='N/A', index=False)

# División train y test entre 70% y 30% respectivamente
train, test = train_test_split(df_nitrato_pred, test_size=0.3, random_state=12345)

# Eliminar la variable dependiente en el set de testeo
test.drop(columns=['ValorNitrato'], inplace=True)

# Print shape y head
print('Train shape:', train.shape)
print('Test shape:', test.shape)
train.head()

# Numero de valores missing por columna
na_df = pd.DataFrame([train.isna().sum(),test.isna().sum()]).T
na_df.columns = ["Train","Test"]
pd.set_option('display.max_rows', None)
na_df

# Tipo de valores de las colu
train.dtypes

# Numero de valores unicos por columna
nu_df = pd.DataFrame([train.nunique(),test.nunique()]).T
nu_df.columns = ["Train","Test"]
nu_df

# Distribucion de la variable dependiente
plt.figure(figsize=(12,4))
sns.histplot(data=train, x="ValorNitrato")
plt.title("Target distribution")
plt.show()

# Maximo y minimo de la variable dependiente
print("Target maximum:", train["ValorNitrato"].max())
print("Target minimum:", train["ValorNitrato"].min())

# Label: Variable dependiente
y = train['ValorNitrato']

# Features: Variables independientes
X = train.drop('ValorNitrato', axis=1)
X_test = test

# One-hot encode para variables independientes categoricas
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Escalar o estandarizar los datos 
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Division de los datos de entrenamiento a entrenamiento y validacion entre 80% y 20%
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12345, shuffle=True)

# Print tamaño del set de entrenamiento y validacion del conjunto de variables independientes
print("X_train shape:", X_train.shape)
print("X_val shape:", X_valid.shape)

# Reset de los indices
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)

# Algoritmos de Machine Learning
regressors = {
    "LinearRegression" : LinearRegression(),
    "SVM" : SVR(),
    "RandomForest" : RandomForestRegressor(random_state=0),
    "XGBoost" : XGBRegressor(random_state=0, use_label_encoder=False, eval_metric='rmse'),
    "LGBM": lgb.LGBMRegressor(random_state=0),
    "CatBoost" : CatBoostRegressor(random_state=0, verbose=False),
    "Red": MLPRegressor()
}

# Grids para el tuneo de hiperparametros
LR_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

SVM_grid = {'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']}

RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, 12]}

boosted_grid = {'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.15]}

Red_grid = {"hidden_layer_sizes": [(5,),(50,),(100,),(150,)], 
            "activation": ["identity", "logistic", "tanh", "relu"], 
            "solver": ["lbfgs", "sgd", "adam"], 
            "alpha": [0.0001, 0.001, 0.01, 0.1]
}

# Diccionario de los grids
grid = {
    "LinearRegression" : {},
    "SVM" : SVM_grid,
    "RandomForest" : RF_grid,
    "XGBoost" : boosted_grid,
    "LGBM": boosted_grid,
    "CatBoost" : boosted_grid,
    "Red": Red_grid
}

# Entrenamiento, modelizacion y evaluacion de los distintos algoritmos
for key, regressor in regressors.items():
    # Start timer
    start = time.time()
    
    # Tune hyperparameters
    reg = GridSearchCV(estimator=regressor, param_grid=grid[key], scoring='neg_root_mean_squared_error', n_jobs=-1, cv=5)
    
    # Train using PredefinedSplit
    reg.fit(X_train, y_train)
    
    # Validation set predictions
    val_preds = reg.predict(X_valid)
    val_preds[val_preds < 0] = 0
    mae = mean_absolute_error(y_valid, val_preds)
    mse = mean_squared_error(y_valid, val_preds)
    rmse = np.sqrt(mse)  # use np.sqrt to calculate RMSE
    r2 = r2_score(y_valid, val_preds)
    
    # Test set predictions
    test_preds = reg.predict(X_test)
    test_preds[test_preds < 0] = 0

    # Stop timer
    stop = time.time()
    
    # Create DataFrames with predictions
    X_valid_with_preds = X_valid.copy()
    X_valid_with_preds['y_valid'] = y_valid
    X_valid_with_preds[f'{key}_val_preds'] = val_preds
    
    X_test_with_preds = X_test.copy()
    X_test_with_preds[f'{key}_test_preds'] = test_preds
    
    # Save predictions to CSV
    X_valid_with_preds.to_csv(f"{key}_valid_preds.csv", index=False)
    X_test_with_preds.to_csv(f"{key}_test_preds.csv", index=False)
    
    # Save valid preds
    pd.DataFrame({"result":val_preds}).to_csv(f"{key}_val_preds.csv", index=False)
        
    # Print score and time
    print('Model:', key)
    print('Validation MAE:', mae)
    print('Validation MSE:', mse)
    print('Validation RMSE:', rmse)
    print('Validation R^2:', r2)
    print('Training time (mins):', np.round((stop - start)/60, 2))
    print('')
    
    # Save variable importance ranking if available
    if hasattr(reg.best_estimator_, 'feature_importances_'):
        importances = reg.best_estimator_.feature_importances_
    elif hasattr(reg.best_estimator_, 'coef_'):
        importances = reg.best_estimator_.coef_
    else:
        importances = None
    
    if importances is not None:
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Print the feature importances
        print(feature_importance_df)
        
        # Save feature importances to CSV
        feature_importance_df.to_csv(f"{key}_feature_importances.csv", index=False)