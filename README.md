AVANCE 1
Lo primero que haremos en el arranque del Proyecto Integrador sera la carga de las librerias, la lectura y carga del archivo data_latinoamerica.csv, y lo haremos de igual forma en los siguientes avance a medida que se vayan incorporando otras librerias utiles para el siguiente analisis.
! pip install seaborn
! pip freeze > requirements.txt
#Cargamos librerias 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#Realizamos la carga de la data con su ruta correspondiente
data = pd.read_csv ("C:\\VSC Gerardo\\data_latinoamerica.csv")
data.head()
#Se realiza chequeo de cantidad de columnas y filas en la data 
np.shape(data)
Se comienza a realizar filtrado de la data, seleccionando los países donde se expandirán: Colombia, Argentina, Chile, México, Perú y Brasil.
paises_seleccionados = ['Argentina', 'Chile', 'Colombia', 'Mexico', 'Peru', 'Brazil']
data_latinoamerica = data[data['country_name'].isin(paises_seleccionados)]
Revisamos la data y veamos cuanto redujo en cantidad de datos
np.shape(data_latinoamerica)
Revisaremos por columnas y generamos una mascara para filtrar cuando los valores faltantes superan los 4millones 
data_latinoamerica.isnull().sum()[data_latinoamerica.isnull().sum()>4000000]
El analisis anterior nos permite ver que hay una cantidad importante de datos faltantes 
Realizaremos filtrado a la columna location_key
data_latinoamerica_paises = data_latinoamerica[data_latinoamerica['location_key'].isin(['AR', 'CL', 'CO', 'MX', 'PE', 'BR'])]
np.shape(data_latinoamerica_paises)
Se filtraran los datos en fechas mayores a 2021-01-01
data_paisesLA_Fecha = data_latinoamerica_paises[data_latinoamerica_paises['date']>'2021-01-01']
Se revisara por filtro los vacios y luego se procedera a eliminar
chek_nulos = data_paisesLA_Fecha.isnull().sum()[data_paisesLA_Fecha.isnull().sum()>0]
print (chek_nulos)
print (np.shape(data_paisesLA_Fecha))
#Guardamos la data ya filtrada
data_paisesLA_Fecha.to_csv('DatosFinalesFiltrado.csv', index=False)

Verificamos columnas numericas 
columnas_numericas = data_paisesLA_Fecha.select_dtypes(include=[np.number])
print (columnas_numericas)
Generamos Bucle para explorar estadisticas y metricas importantes para el analisis, evaluando por cada columna 
# Bucle for para recorrer las columnas numéricas
for columna in columnas_numericas:
    # Cálculo de métricas con `describe()`
    descriptivas = data_paisesLA_Fecha[columna].describe()

    # Impresión de métricas
    print(f"\n**Métricas descriptivas para la columna {columna}:**")
    print(descriptivas)
# Análisis de la media y la desviación estándar
media = descriptivas["mean"]
desviacion_estandar = descriptivas["std"]

print(f"\n**Análisis de media y desviación estándar:**")
print(f"La media de la columna {columna} es: {media}")
print(f"La desviación estándar de la columna {columna} es: {desviacion_estandar}")

AVANCE 2
#IMPORTAMOS LAS LIBRERIAS PARA TRABAJAR EN ESTE AVANCE 
import pandas as pd 
import matplotlib.pyplot as plt
! pip install seaborn 
import seaborn as sns
import numpy as np
En este avance cargamos la data ya filtrada que sera con la que trabjaremos el resto del Proyecto Integrador 
data_final_filtrada = pd.read_csv("C:\\VSC Gerardo\\DatosFinalesFiltrado.csv")
print(data_final_filtrada)
#Convirtiendo la columna date en formato datetime
pd.to_datetime(data_final_filtrada['date'])
data_final_filtrada['date'].dtype
Comenzaremos a generar los primeros graficos basados en nuestro df filtrado, que permitira ir viendo y extrayendo insigth valiosos 
import matplotlib.dates as mdates

# Incidencia acumulada de COVID-19
plt.figure(figsize=(14, 7))
sns.lineplot(data_final_filtrada, x='date', y='cumulative_confirmed', hue='country_name')
plt.title('Incidencia acumulada de COVID-19 por país')
plt.xlabel('Fecha')
plt.ylabel('Casos confirmados acumulados')
plt.legend(title='País')

# Formatear las etiquetas del eje X para mostrar solo los años
plt.gca().xaxis.set_major_locator(mdates.YearLocator())


plt.show()
El grafico generado aca arriba su primera impresion nos detalla como fue la evolucion de casos confirmados (y que posteriormente podemos hacer el comparativo con el grafico de vacunados) y ver su desarrollo en tiempo de los principales paises de latinoamerica, considerando por supuesto a Brazil quien marca tendencia en el grafico producto de su alta densidad poblacional
# Tasa de vacunación acumulada
plt.figure(figsize=(14, 7))
sns.lineplot(data_final_filtrada, x='date', y='cumulative_vaccine_doses_administered', hue='country_name')
plt.title('Dosis de vacunas administradas acumuladas por país')
plt.xlabel('Fecha')
plt.ylabel('Dosis de vacunas administradas')
plt.legend(title='País')

# Formatear las etiquetas del eje X para mostrar solo los años
plt.gca().xaxis.set_major_locator(mdates.YearLocator())


plt.show()
Comparando con el primer grafico generado, este nos permite ver como el desarrollo de la vacunacion en la poblacion permitio el estancamiento de los aumentos de casos confirmados en algunos paises 
# Tasa de vacunación acumulada
plt.figure(figsize=(14, 7))
sns.lineplot(data_final_filtrada, x='date', y='cumulative_deceased', hue='country_name')
plt.title('Muertes acumuladas por país')
plt.xlabel('Fecha')
plt.ylabel('Muertes Acumuladas')
plt.legend(title='País')

# Formatear las etiquetas del eje X para mostrar solo los años
plt.gca().xaxis.set_major_locator(mdates.YearLocator())


plt.show()
Este grafico permite observar como los fallecidos por COVID fueron mermando en funcion de los procesos de vacunacion en latinoamerica, llamado la atencion el caso de Mexico donde pasado un año de sus planes de vacunacion las muertes acumuladas en ese paises mermaron casi el triple en sus indices estadisticos.
# Histograma y gráfico de densidad de incidencia de COVID-19
plt.figure(figsize=(12, 6))
sns.histplot(data_final_filtrada, x='cumulative_confirmed', kde=True)
plt.title('Distribución de la incidencia de COVID-19')
plt.xlabel('Casos confirmados acumulados')
plt.ylabel('Frecuencia')
plt.show()
# Histograma y gráfico de densidad de tasas de vacunación
plt.figure(figsize=(12, 6))
sns.histplot(data_final_filtrada, x='cumulative_vaccine_doses_administered', kde=True)
plt.title('Distribución de las tasas de vacunación')
plt.xlabel('Dosis de vacunas administradas acumuladas')
plt.ylabel('Frecuencia')
plt.show()
# Gráfico de barras para comparar la incidencia de COVID-19 entre países
plt.figure(figsize=(14, 7))
sns.barplot(data_final_filtrada, x='country_name', y='cumulative_confirmed')
plt.title('Comparación de la incidencia de COVID-19 entre países')
plt.xlabel('País')
plt.ylabel('Casos confirmados acumulados')
plt.xticks(rotation=90)
plt.show()
Visual comparativa de casos confirmados entre los paises 
# Mapa de calor de correlación
numeric_columns = data_final_filtrada.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data_final_filtrada[numeric_columns].corr()
mask= np.triu(np.ones_like(correlation_matrix, dtype=bool))
correlation_matrix = correlation_matrix[abs(correlation_matrix) > abs(0.5)]
plt.figure(figsize=(50, 30))
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt="1f", annot_kws={"size": 12})
plt.title('Mapa de calor de correlación', fontsize=50)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=20)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20)
plt.show()

# Diagrama de dispersión de la temperatura media contra los casos confirmados
plt.figure(figsize=(12, 6))
sns.scatterplot(data_final_filtrada, x='average_temperature_celsius', y='cumulative_confirmed', hue='country_name')
plt.title('Temperatura media vs. Casos confirmados')
plt.xlabel('Temperatura media (°C)')
plt.ylabel('Casos confirmados acumulados')
plt.show()
En el anterior diagrama podemos notar como bajo oscilacion de 20º y 30º de Temp se concentra la mayor cantidad de casos confirmados 
# Diagrama de dispersión de la temperatura media contra las muertes confirmadas
plt.figure(figsize=(12, 6))
sns.scatterplot(data_final_filtrada, x='average_temperature_celsius', y='cumulative_deceased', hue="country_name")
plt.title('Temperatura media vs. Muertes confirmadas')
plt.xlabel('Temperatura media (°C)')
plt.ylabel('Muertes confirmadas acumuladas')
plt.show()
El pick de las 250mil muertes el diagrama lo enfatizo en una Temperatura promedio de 25º
# Comportamiento de las dosis administradas de todos los países (Valor medio)
mean_doses = data_final_filtrada.groupby('country_name')['cumulative_vaccine_doses_administered'].mean().reset_index()
plt.figure(figsize=(14, 7))
sns.barplot(data=mean_doses, x='country_name', y='cumulative_vaccine_doses_administered')
plt.title('Valor medio de dosis administradas por país')
plt.xlabel('País')
plt.ylabel('Dosis de vacunas administradas (media)')
plt.xticks(rotation=90)
plt.show()
El grafico de barra anterior destaca a Brasil y Mexico como quienes mas aplicaron vacunas pero vale recordar que en el los datos esta influyendo por supuesto la densidad demografica, por lo cual observando a Peru, Colombia, Chile y Argentina tuvieron similitud en las dosis aplicadas

#Se Crea la columna 'month' extrayendo el mes de la columna 'date'
data_final_filtrada['month'] = data_final_filtrada['date'].dt.month

#Verificamos que la columna 'month' se haya creado correctamente
print(data_final_filtrada[['date', 'month']])

# Crea un gráfico de líneas utilizando la columna 'month'
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_final_filtrada, x='date', y='cumulative_vaccine_doses_administered', hue='country_name')
plt.title('Evolución de dosis administradas por mes')
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [label.replace('country_name', 'PAIS') for label in labels]
plt.legend(handles, new_labels, title='PAIS')
plt.show()


De nuestra data  general ejecutamos un grafico de linea para visualizar la evolucion en la administracion de dosis por mes, siendo este mas especifico por mes 


#Nos aseguramos que la columna 'date' esté en formato datetime
data_final_filtrada['date'] = pd.to_datetime(data_final_filtrada['date'], errors='coerce')

#Creamos la columna 'month' a partir de la columna 'date'
data_final_filtrada['month'] = data_final_filtrada['date'].dt.to_period('M')

#Se verifica los primeros registros para asegurar que 'month' se creó correctamente
print(data_final_filtrada[['date', 'month', 'cumulative_vaccine_doses_administered']].head())

#Verificamos si hay valores nulos en 'month' o 'cumulative_vaccine_doses_administered'
print(data_final_filtrada[['month', 'cumulative_vaccine_doses_administered']].isnull().sum())

#Nos Aseguramos de que 'cumulative_vaccine_doses_administered' es numérico
data_final_filtrada['cumulative_vaccine_doses_administered'] = pd.to_numeric(data_final_filtrada['cumulative_vaccine_doses_administered'], errors='coerce')

#Se vuelve a verificar si hay valores nulos después de la conversión
print(data_final_filtrada[['cumulative_vaccine_doses_administered']].isnull().sum())




#Agrupamos los datos por mes y país y sumar las dosis administradas
doses_by_month = data_final_filtrada.groupby(['country_name', 'month'])['cumulative_vaccine_doses_administered'].sum().reset_index()


print(doses_by_month.head())

La ejecucion de los codigo anteriores se realizo ya que estamos presentando error de formarto especificamente en le columna creada Month
import seaborn as sns
import matplotlib.pyplot as plt

# Convertir 'month' a formato de fecha si es necesario para la visualización
doses_by_month['month'] = doses_by_month['month'].dt.to_timestamp()

# Crear el gráfico de líneas
plt.figure(figsize=(14, 7))
sns.lineplot(data=doses_by_month, x='month', y='cumulative_vaccine_doses_administered', hue='country_name')
plt.title('Evolución de dosis administradas por mes de cada país')
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [label.replace('country_name', 'PAIS') for label in labels]
plt.legend(handles, new_labels, title='PAIS')
plt.xlabel('Mes')
plt.ylabel('Dosis acumuladas administradas')
plt.xticks(rotation=45)  # Opcional: Rote las etiquetas para mejorar la legibilidad
plt.show()

Vemos como cada pais fue incrementando sus dosis administrada, siempre en alza con leves periodos de estancamiento 
print(data_final_filtrada[['date', 'cumulative_vaccine_doses_administered']].isnull().sum())
# Extraer el mes y año de la columna 'date'
data_final_filtrada['month'] = data_final_filtrada['date'].dt.to_period('M')

# Agrupar por mes y sumar las dosis administradas
monthly_vaccine_doses = data_final_filtrada.groupby('month')['cumulative_vaccine_doses_administered'].sum().reset_index()

# Convertir la columna 'month' a datetime para que Seaborn pueda interpretarla correctamente
monthly_vaccine_doses['month'] = monthly_vaccine_doses['month'].dt.to_timestamp()

# Verificar el DataFrame resultante
print(monthly_vaccine_doses.head())
# Llenar o eliminar valores nulos si es necesario
data_final_filtrada = data_final_filtrada.dropna(subset=['date', 'cumulative_vaccine_doses_administered'])
# Agrupar por mes y país y sumar las muertes
monthly_deceased = data_final_filtrada.groupby(['month', 'country_name'])['cumulative_deceased'].sum().reset_index()
monthly_confirmed = data_final_filtrada.groupby(['month', 'country_name'])['cumulative_confirmed'].sum().reset_index()
monthly_recovered = data_final_filtrada.groupby(['month', 'country_name'])['cumulative_recovered'].sum().reset_index()

# Convertir la columna 'month' a datetime para que Seaborn pueda interpretarla correctamente
monthly_deceased['month'] = monthly_deceased['month'].dt.to_timestamp()
monthly_confirmed['month'] = monthly_confirmed['month'].dt.to_timestamp()
monthly_recovered['month'] = monthly_recovered['month'].dt.to_timestamp()
Lo codigos anteriores fueron necesarios para poder ejecutar los codigos posteriores y poder tener una buena visual de los graficos 
# Gráfico de Muertes por Mes de Cada País
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_deceased, x='month', y='cumulative_deceased', hue='country_name')
plt.title('Muertes por Mes de Cada País')
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [label.replace('country_name', 'PAIS') for label in labels]
plt.legend(handles, new_labels, title='PAIS')
plt.xlabel('Fecha')
plt.ylabel('Muertes Acumuladas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de Casos Confirmados por Mes de Cada País
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_confirmed, x='month', y='cumulative_confirmed', hue='country_name')
plt.title('Casos Confirmados por Mes de Cada País')
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [label.replace('country_name', 'PAIS') for label in labels]
plt.legend(handles, new_labels, title='PAIS')
plt.xlabel('Fecha')
plt.ylabel('Casos Confirmados Acumulados')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de Recuperaciones por Mes de Cada País
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_recovered, x='month', y='cumulative_recovered', hue='country_name')
plt.title('Recuperaciones por Mes de Cada País')
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [label.replace('country_name', 'PAIS') for label in labels]
plt.legend(handles, new_labels, title='PAIS')
plt.xlabel('Fecha')
plt.ylabel('Recuperaciones Acumuladas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
Con lo anterior y considerando la mayoria de los paises todo matuvieron la linea media durante el paso del tiempo, en el primer grafico observamos como Brasil siempre su tendencia fue en aumento mientras que Mexico desde el tercer trimestre del 2021 fue en descenso, d eigula forma en el segundo grafico todos los paises se mantuvieron en el tiempo mientras que Brasil siempre fue en ascenso y el ultimo grafico Brasil y Colombia tuvieron mas recuperados mientras que le resto de los paises se mantuvieron estaticos 
# Comparación del número de casos nuevos entre países
plt.figure(figsize=(14, 7))
sns.lineplot(data=data_final_filtrada, x='date', y='new_confirmed', hue='country_name')
plt.title('Comparación del número de casos nuevos entre países')
plt.xlabel('Fecha')
plt.ylabel('Casos nuevos')
plt.legend(title='País')
plt.show()
# Dosis acumuladas por país
plt.figure(figsize=(14, 7))
sns.barplot(data=data_final_filtrada, x='country_name', y='cumulative_vaccine_doses_administered')
plt.title('Dosis acumuladas por país')
plt.xlabel('País')
plt.ylabel('Dosis de vacunas administradas acumuladas')
plt.xticks(rotation=90)
plt.show()
# Boxplot de temperatura media de cada país
plt.figure(figsize=(14, 7))
sns.boxplot(data=data_final_filtrada, x='country_name', y='average_temperature_celsius')
plt.title('Boxplot de temperatura media de cada país')
plt.xlabel('País')
plt.ylabel('Temperatura media (°C)')
plt.xticks(rotation=90)
plt.show()
Temperatura media por cada pais 
# Violinplot de las variables con cambios de valores (CASOS CONFIRMADOS ACUMULADOS)
plt.figure(figsize=(14, 7))
sns.violinplot(data=data_final_filtrada, x='country_name', y='cumulative_confirmed')
plt.title('Violinplot de casos confirmados acumulados por país')
plt.xlabel('País')
plt.ylabel('Casos confirmados acumulados')
plt.xticks(rotation=90)
plt.show()
Claramente el violinplot se ve afectado por los datos de Brasil al igual que lo otros graficos, sin embargo tomando en cuenta el resto de los paises observamos con la diferencia en la simetria es baja  
# Violinplot de las variables con cambios de valores (MUERTES ACUMULADAS)
plt.figure(figsize=(14, 7))
sns.violinplot(data=data_final_filtrada, x='country_name', y='cumulative_deceased')
plt.title('Violinplot de casos confirmados acumulados por país')
plt.xlabel('País')
plt.ylabel('Casos confirmados acumulados')
plt.xticks(rotation=90)
plt.show()
La media no tiene simetria 
for col in data_final_filtrada_6.columns:
    if col not in ['location_key', 'date', 'country_code', 'latitude', 'longitude', 'country_name']:
        # Forzar conversión a numérico, con coerción de errores
        data_final_filtrada_6[col] = pd.to_numeric(data_final_filtrada_6[col], errors='coerce')

data_final_filtrada_6 = data_final_filtrada_6.dropna()
data_final_filtrada_6 = data_final_filtrada_6.fillna(0)

# Forzar conversión a tipo numérico
data_final_filtrada_6[i] = pd.to_numeric(data_final_filtrada_6[i], errors='coerce')

# Elimina filas con valores NaN en la columna
data_final_filtrada_6 = data_final_filtrada_6.dropna(subset=[i])


print(data_final_filtrada_6.head(5))
# Se crea un dataframe con las columnas mayores a 6 porque son las que tienen valores diferentes para cada pais. 
columnas_mayores_6=[] 
for i in data_final_filtrada.columns: 
    if i not in ['location_key', 'date', 'country_code', 'latitude', 'longitude']: 
        if data_final_filtrada[i].nunique()>6: 
            columnas_mayores_6.append(i)
#agregar a columnas_mayores_6 la columna' country_name para poder hacer analisis por pais. 
columnas_mayores_6.append('country_name') 
# Se crea un dataframe con las columnas mayores a 6 
data_final_filtrada_6=data_final_filtrada[columnas_mayores_6]

for i in data_final_filtrada_6.columns:
    if i not in ['location_key', 'date', 'country_code', 'latitude', 'longitude', 'country_name']:
        # Check if the current column (i) is numeric
        if pd.api.types.is_numeric_dtype(data_final_filtrada_6[i]):
            plt.figure(figsize=(12, 8))
            sns.violinplot(x='country_name', y=i, data=data_final_filtrada_6, hue='country_name')
            plt.title(f'Distribución de {i} por País en Latinoamérica')
            plt.xticks(rotation=45)  # Optional: Rotate labels for readability
            plt.show()
        else:
            print(f"Warning: Column '{i}' is not numeric. Skipping violin plot.")
LOS VIOLINPLOTS PRESENTADO RESPONDE A UN EJERCICO PARA VISUALIZAR PARTE DE LOS GRAFICOS YA PRESENTADOS ANTERIORMENTE PERO AHORA VISUALIZADOS COMO VIOLINPLOT QUE NOS PERMITE VER LAS VARIACIONES DE LAS LINEAS MEDIAS EN LOS DISTINTOS DATOS 
# Distribución de la población por grupos de edad
plt.figure(figsize=(14, 7))
age_groups = ['population_age_70_79', 'population_age_80_and_older', 'population_age_00_09', 'population_age_10_19', 'population_age_20_29', 'population_age_30_39', 'population_age_40_49', 'population_age_50_59', 'population_age_60_69']
data_age = data_final_filtrada.melt(id_vars='country_name', value_vars=age_groups, var_name='Age Group', value_name='Population')
sns.barplot(data=data_age, x='country_name', y='Population', hue='Age Group')
plt.title('Distribución de la población por grupos de edad')
plt.xlabel('País')
plt.ylabel('Población')
plt.xticks(rotation=90)
plt.show()
DISTRIBUCION POBLACIONAL 
# Mapa de calor de métricas por país
metrics = ['cumulative_confirmed', 'cumulative_deceased', 'cumulative_recovered', 'cumulative_vaccine_doses_administered']
data_metrics = data_final_filtrada.groupby('country_name')[metrics].sum().reset_index().set_index('country_name')
plt.figure(figsize=(14, 7))
sns.heatmap(data_metrics, annot=True, cmap='YlGnBu', fmt='1.0f')
plt.title('Mapa de calor de métricas por país')
plt.xticks(rotation=45)
plt.show()
data_final_filtrada.isna()
data_final_filtrada.info()

AVANCE 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
data_final_filtrada = pd.read_csv("C:\\VSC Gerardo\\DatosFinalesFiltrado.csv")
data_final_filtrada
INVOCAMOS NUESTRA DATA FILTRADA
data_final_filtrada['date'] = pd.to_datetime(data_final_filtrada['date'])
data_final_filtrada = data_final_filtrada.set_index("date")
data_final_filtrada
TRAEMOS EL DATE COMO INDICE 
data_final_filtrada["week"] = data_final_filtrada.index.isocalendar().week
data_final_filtrada
data_final_filtrada["YEAR"] = data_final_filtrada.index.isocalendar().year
data_final_filtrada
print(data_final_filtrada)
CREAMOS LA COLUMNA SEMANA 
data_semanal = data_final_filtrada.resample('W').sum() # O usa .mean() si quieres promedios


plt.figure(figsize=(14, 7))
sns.lineplot(data=data_semanal, x=data_semanal.index, y='new_confirmed')
plt.title('Evolución de casos de forma semanal')
plt.xlabel('Fecha')
plt.ylabel('Casos acumulados')
plt.show()
data_mensual = data_final_filtrada.resample('M').sum()  # O usa .mean() si quieres promedios

# Graficar evolución mensual
plt.figure(figsize=(14, 7))
sns.lineplot(data=data_mensual, x=data_mensual.index, y='new_confirmed')
plt.title('Evolución de casos de forma mensual')
plt.xlabel('Fecha')
plt.ylabel('Casos acumulados')
plt.show()
# Resampleo trimestral
quarterly_data = data_final_filtrada.resample('Q').sum()  # O usa .mean() si quieres promedios

# Graficar evolución trimestral
plt.figure(figsize=(14, 7))
sns.lineplot(data=quarterly_data, x=quarterly_data.index, y='new_confirmed')
plt.title('Evolución de casos de forma trimestral')
plt.xlabel('Fecha')
plt.ylabel('Casos acumulados')
plt.show()
OBSERVAMOS EL COMPORTAMIENTO DE CASOS GLOBALES DE FORMA SEMANAL, QUINCENAL Y TRIMESTRAL 
casos_semanales['country_name'].unique()
weekly_cases = data_final_filtrada.groupby(['country_name', 'week'])[['new_confirmed', 'new_deceased' ]].sum().reset_index() # type: ignore 

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 16)) 
for country in weekly_cases['country_name'].unique(): 
    country_weekly_cases = weekly_cases[weekly_cases['country_name'] == country] 
    ax[0].plot(country_weekly_cases ['week'], country_weekly_cases ['new_confirmed'], label=f'{country} ') 
    ax[1].plot(country_weekly_cases['week'], country_weekly_cases['new_deceased'], label=f'{country} ') 

ax[0].set_xlabel("Semana del Año") 
ax[0].set_ylabel("Casos Nuevos de COVID-19") 
ax[0].set_title("Evolución Semanal de Casos Nuevos de COVID-19 en América Latina") 
ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

ax[1].set_xlabel("Semana del Año") 
ax[1].set_ylabel ("Muertes de COVID-19") 
ax[1].set_title("Evolución Semanal de Muertes por COVID-19 en América Latina") 
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') 

plt.tight_layout() 
plt.show()

GENERAMOS 2 GRAFICOS PARA VER LA EVOILUCION SEMANAL DE CASOS Y MUERTES DETALLADOS POR PAIS, ALLI VEMOS COMO LOS PICK MAS ALTO SE CONCENTRAN EN EL INCIO DE LA PANDEMIA Y PRODUCTO DE LAS DISTINTAS POLITICAS DE SALUD FUE DISMINUYENDO 
data_final_filtrada["month"] = data_final_filtrada.index.month
data_final_filtrada
yearly_cases = data_final_filtrada.groupby(['country_name', 'month'])[['new_confirmed', 'new_deceased' ]].sum().reset_index()

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 16)) 
for country in yearly_cases['country_name'].unique(): 
    country_yearly_cases = yearly_cases[yearly_cases['country_name'] == country] 
    ax[0].plot(country_yearly_cases ['month'], country_yearly_cases ['new_confirmed'], label=f'{country} ') 
    ax[1].plot(country_yearly_cases['month'], country_yearly_cases['new_deceased'], label=f'{country} ') 

ax[0].set_xlabel("Mes del Año 2021") 
ax[0].set_ylabel("Casos Nuevos de COVID-19") 
ax[0].set_title("Evolución Mensual de Casos Nuevos de COVID-19 en América Latina") 
ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

ax[1].set_xlabel("Mes del Año 2021") 
ax[1].set_ylabel ("Muertes de COVID-19") 
ax[1].set_title("Evolución Mensual de Muertes por COVID-19 en América Latina") 
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') 

plt.tight_layout() 
plt.show()
ESTOS DOS NOS PERMITE DETALLAR IGUAL QUE LOS ANTERIORES PERO ESTOS DE FORMA MENSUAL
data_final_filtrada
data_final_filtrada
# Restablece el índice si 'date' está como índice
if 'date' in data_final_filtrada.index.names:
    data_final_filtrada = data_final_filtrada.reset_index()

# Asegúrate de que la columna 'date' esté en formato datetime
data_final_filtrada['date'] = pd.to_datetime(data_final_filtrada['date'])
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_final_filtrada, x='date', y='new_confirmed', hue='country_name')
plt.title('Evolución de Casos Nuevos Confirmados')
plt.xlabel('Fecha')
plt.ylabel('Número de Casos')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_final_filtrada, x='date', y='cumulative_confirmed', hue='country_name', linestyle='--')
plt.title('Evolución de Casos Confirmados')
plt.xlabel('Fecha')
plt.ylabel('Número de Casos')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_final_filtrada, x='date', y='cumulative_vaccine_doses_administered', hue='country_name')
plt.title('Progreso de la vacunación por país')
plt.xlabel('Fecha')
plt.ylabel('Dosis de Vacunas Administradas')
plt.show()

data_final_filtrada['growth_rate'] = data_final_filtrada['new_confirmed'].pct_change() * 100

plt.figure(figsize=(12, 6))
sns.lineplot(data=data_final_filtrada, x='date', y='growth_rate', hue='country_name')
plt.title('Tasa de Crecimiento (%)')
plt.xlabel('Fecha')
plt.ylabel('Tasa de Crecimiento (%)')
plt.show()
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data_final_filtrada, x='cumulative_vaccine_doses_administered', y='new_confirmed', hue='country_name')
plt.title('Relación entre la Cobertura de Vacunación y la Reducción de Casos')
plt.xlabel('Dosis de Vacunas Administradas')
plt.ylabel('Nuevos Casos')
plt.show()
plt.figure(figsize=(15, 10))

sns.lineplot(data=data_final_filtrada, x='date', y='average_temperature_celsius', hue='country_name', linestyle='--')
plt.title('Nuevos casos y temperatura promedio')
plt.xlabel('Fecha')
plt.ylabel('Número de Casos / Temperatura')
plt.legend()
plt.show()
data_vaccination = data_final_filtrada.groupby('country_name')['cumulative_vaccine_doses_administered'].max().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=data_vaccination.values, y=data_vaccination.index, hue=data_vaccination.index)
plt.xlabel('Dosis de vacunas administradas')
plt.ylabel('Pais')
plt.title("Progreso de vacunacion por pais")
plt.show()
plt.figure(figsize=(12, 6))
sns.barplot(data=data_final_filtrada, x='country_name', y='cumulative_vaccine_doses_administered')
plt.title('Comparación de Estrategias de Vacunación en América Latina')
plt.xlabel('País')
plt.ylabel('Dosis de Vacunas Administradas')
plt.show()
plt.figure(figsize=(12, 6))
sns.barplot(data=data_final_filtrada, x='cumulative_deceased', y='country_name', hue='country_name')

plt.title('Evolución del Número de Muertes Diarias')
plt.xlabel('Cantidad Muertos')
plt.ylabel('Paises')
plt.show()
# Asumimos que data_final_filtrada tiene una columna 'mortality_rate'
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_final_filtrada, x='pollution_mortality_rate', y='comorbidity_mortality_rate', hue='country_name')
plt.title('Prevalencia de Condiciones Preexistentes en Países con Altas y Bajas Tasas de Mortalidad')
plt.xlabel('Tasa de Mortalidad')
plt.ylabel('Condiciones Preexistentes')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=data_final_filtrada, x='date', y='cumulative_deceased', hue='country_name')
plt.title('Análisis Temporal de la Mortalidad')
plt.xlabel('Fecha')
plt.ylabel('Número de Muertes')
plt.show()
ultimo_mes = data_final_filtrada['date'].max() - pd.DateOffset(months=1)
situacion_actual = data_final_filtrada[data_final_filtrada['date'] >= ultimo_mes]

plt.figure(figsize=(12, 6))
sns.barplot(data=situacion_actual, x='country_name', y='new_confirmed')
plt.title('Comparación de la Situación Actual')
plt.xlabel('País')
plt.ylabel('Nuevos Casos en el Último Mes')
plt.show()
# Asegurarse de que la columna 'date' sea de tipo datetime
data_final_filtrada['date'] = pd.to_datetime(data_final_filtrada['date'])

# Establecer la columna 'date' como índice
data_final_filtrada.set_index('date', inplace=True)
plt.figure(figsize=(20, 7))

ax1 = plt.gca()  # Obtiene el eje actual de la gráfica y lo asigna a ax1.
ax2 = ax1.twinx()  # Crea un segundo eje ax2 que comparte el mismo eje x que ax1, permitiendo graficar dos conjuntos de datos con diferentes escalas y unidades.

# Graficar los nuevos casos confirmados.
ax1.plot(data_final_filtrada.resample('M').mean(numeric_only=True).index,
         data_final_filtrada.resample('M').mean(numeric_only=True)['new_confirmed'],
         color='red', label='Nuevos casos confirmados')

# Graficar la temperatura promedio.
ax2.plot(data_final_filtrada.resample('M').mean(numeric_only=True).index,
         data_final_filtrada.resample('M').mean(numeric_only=True)['average_temperature_celsius'],
         color='blue', label='Temperatura promedio (°C)')

# Etiquetas de los ejes
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Nuevos casos confirmados', color='red')
ax2.set_ylabel('Temperatura promedio (°C)', color='blue')

# Título de la gráfica
plt.title('Nuevos casos de COVID-19 y temperatura promedio con el tiempo')

# Leyendas
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Mostrar la gráfica
plt.show()
plt.figure(figsize=(20, 7))

ax1 = plt.gca()  # Obtiene el eje actual de la gráfica y lo asigna a ax1.
ax2 = ax1.twinx()  # Crea un segundo eje ax2 que comparte el mismo eje x que ax1, permitiendo graficar dos conjuntos de datos con diferentes escalas y unidades.

# Graficar los nuevos casos confirmados.
ax1.plot(data_final_filtrada.resample('M').mean(numeric_only=True).index,
         data_final_filtrada.resample('M').mean(numeric_only=True)['new_confirmed'],
         color='blue', label='Nuevos casos confirmados')

# Graficar la temperatura promedio.
ax2.plot(data_final_filtrada.resample('M').mean(numeric_only=True).index,
         data_final_filtrada.resample('M').mean(numeric_only=True)['new_deceased'],
         color='red', label='Nuevas Muertes')

# Etiquetas de los ejes
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Nuevos casos confirmados', color='red')
ax2.set_ylabel('Nuevas Muertes', color='blue')

# Título de la gráfica
plt.title('Nuevos casos de COVID-19 VS Muertes')

# Leyendas
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Mostrar la gráfica
plt.show()
pip install folium

import pandas as pd
import folium

# Cargar el dataset
data_final_filtrada = pd.read_csv("C:\\VSC Gerardo\\DatosFinalesFiltrado.csv")

# Verificar y limpiar los datos
data_final_filtrada = data_final_filtrada.dropna(subset=['country_name', 'cumulative_vaccine_doses_administered', 'latitude', 'longitude'])

# Asegúrate de que las columnas 'cumulative_deceased' y 'latitude', 'longitude' son del tipo correcto
data_final_filtrada['cumulative_vaccine_doses_administered'] = data_final_filtrada['cumulative_vaccine_doses_administered'].astype(float)
data_final_filtrada['latitude'] = data_final_filtrada['latitude'].astype(float)
data_final_filtrada['longitude'] = data_final_filtrada['longitude'].astype(float)

# Crear un mapa centrado en el centro geográfico de tus datos
map_center = [data_final_filtrada['latitude'].mean(), data_final_filtrada['longitude'].mean()]
mapa = folium.Map(location=map_center, zoom_start=5)

# Añadir marcadores al mapa
for idx, row in data_final_filtrada.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,  # Tamaño del círculo
        popup=f"País: {row['country_name']}<br>Muertes: {row['cumulative_vaccine_doses_administered']}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(mapa)

# Guardar el mapa en un archivo HTML
mapa.save("mapa_covid.html")

# Mostrar el mapa en un Jupyter Notebook
mapa
