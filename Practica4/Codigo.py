# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Leer conjunto de datos
df = sqlContext.read.csv("/FileStore/tables/carlos_morales_aguilera-training.csv", sep=",", header=True, inferSchema=True)

# Se muestra el dataset
display(df)

# Omisión de valores perdidos
df = df.na.drop()

# Crear columna features
assembler = VectorAssembler(inputCols=df.columns[:-1],outputCol="features")
df = assembler.transform(df)

# Cambiar columna class por label
df = df.withColumnRenamed("class", "label")
drop_cols = ['PSSM_central_-1_I', 'PSSM_r2_-1_R', 'AA_freq_global_S', 'PSSM_central_-2_Y', 'PSSM_central_-1_R', 'PSSM_r1_0_K']
df = df.drop(*drop_cols)
print(df.columns)

# Balancear las clases, para ello filtramos segun clases
df_0 = df.filter("label = 0")
df_1 = df.filter("label = 1")
# Comprobamos que conjunto posee mas elementos
print("Clase 0: " + str(df_0.count()) + "\nClase 1: " + str(df_1.count()))

# Aplicamos downsampling aleatorio sobre el conjunto de la clase 0
df_0_resample = df_0.sample(withReplacement=False, fraction=1.0,seed=0).limit(df_1.count())

# Unimos finalmente el conjunto de entrenamiento
df = df_0_resample.union(df_1)
print(df.count())

# COMMAND ----------

# Indexamos las etiquetas (clase)
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
# Indexamos las caracteristicas (max 4 categorias o sino se considera variables continuas)
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

# Realizar particiones de train (0.8) y test (0.2)
data_train, data_test = df.randomSplit([0.8, 0.2], 0)

# Modelo de Arbol de decision
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=5, maxBins=32)

# Definimos el Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Entrenar modelo
model = pipeline.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))

# COMMAND ----------

# Modelo de Arbol de decision
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=10, maxBins=64)

# Definimos el Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Entrenar modelo
model = pipeline.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))

# COMMAND ----------

# Modelo de RandomForest
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Transformar las etiquetas
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Definimos el Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Entrenar modelo
model = pipeline.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))

# COMMAND ----------

# Modelo de RandomForest
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=25)

# Transformar las etiquetas
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Definimos el Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Entrenar modelo
model = pipeline.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))

# COMMAND ----------

# Modelo de SVM Lineal
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Entrenar modelo
model = lsvc.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = " + str(accuracy))

# COMMAND ----------

# Modelo de SVM Lineal
lsvc = LinearSVC(maxIter=20, regParam=0.3)

# Entrenar modelo
model = lsvc.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = " + str(accuracy))

# COMMAND ----------

# Modelo de Regresion Logistica
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Entrenar modelo
model = lr.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = " + str(accuracy))

# COMMAND ----------

# Modelo de Regresion Logistica
lr = LogisticRegression(maxIter=30, regParam=0.2, elasticNetParam=0.9)

# Entrenar modelo
model = lr.fit(data_train)

# Realizar predicciones
predictions = model.transform(data_test)

# Calcular accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = " + str(accuracy))

# COMMAND ----------


