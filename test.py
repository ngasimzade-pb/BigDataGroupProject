import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

# Initialize Spark
import findspark

findspark.init()

# Import required libraries
import pandas as pd
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StopWordsRemover, RegexTokenizer, CountVectorizer, Word2Vec, \
    NGram
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import VectorUDT
from pyspark.ml.linalg import Vectors

# Start Spark session
sp = (SparkSession.builder
      .appName('Air Quality Data Classification')
      .config("spark.sql.crossJoin.enabled", "true")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate())

sp.sparkContext.setLogLevel('INFO')


# Load Dataset from Local File
def load_csv_to_spark(file_path: str):
    """
    Load a CSV file into a Spark DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - Spark DataFrame
    """
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        pandas_df = pd.read_csv(file_path, sep=';', header=0)
        spark_df = sp.createDataFrame(pandas_df)
        return spark_df
    else:
        raise FileNotFoundError(f"File {file_path} not found!")


# Specify file path
file_path = "./AirQualityUCI.csv"
data_df = load_csv_to_spark(file_path)

# Display Schema and Initial Data
data_df.printSchema()
data_df.show(5)

# Tokenization and Stop Words Removal
tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
wordsData = tokenizer.transform(data_df)

stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
wordsData = stopwords_remover.transform(wordsData)

# TF-IDF Features
hashTF = HashingTF(inputCol="filtered_tokens", outputCol="tfFeatures")
tfData = hashTF.transform(wordsData)

idf = IDF(inputCol="tfFeatures", outputCol="tfidfFeatures")
idfModel = idf.fit(tfData)
tfidfData = idfModel.transform(tfData)

# Bag of Words (BOW) Features
count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="BOWFeatures")
bowData = count_vectorizer.fit(wordsData).transform(wordsData)

# N-Gram Features
n_value = 3
ngram = NGram(n=n_value, inputCol="filtered_tokens", outputCol="ngram_features")
ngramData = ngram.transform(wordsData)

# Word2Vec Features
word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_tokens", outputCol="Word2VecFeatures")
word2vec_model = word2vec.fit(wordsData)
word2vecData = word2vec_model.transform(wordsData)


# Model Evaluation Function
def evaluate_model(predictions, label_col="class_label", prediction_col="prediction"):
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                           metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)
    print(f"Accuracy: {accuracy}")

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="f1")
    f1 = evaluator_f1.evaluate(predictions)
    print(f"F1 Score: {f1}")

    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                         metricName="weightedRecall")
    recall = evaluator_recall.evaluate(predictions)
    print(f"Recall: {recall}")

    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                            metricName="weightedPrecision")
    precision = evaluator_precision.evaluate(predictions)
    print(f"Precision: {precision}")


# Logistic Regression Model
lr = LogisticRegression(regParam=0.1, labelCol="class_label", featuresCol="tfidfFeatures")
lr_pipeline = Pipeline(stages=[lr])
lr_model = lr_pipeline.fit(tfidfData)
lr_predictions = lr_model.transform(tfidfData)
evaluate_model(lr_predictions)

# Decision Tree Model
dt = DecisionTreeClassifier(labelCol="class_label", featuresCol="tfidfFeatures", maxDepth=7)
dt_pipeline = Pipeline(stages=[dt])
dt_model = dt_pipeline.fit(tfidfData)
dt_predictions = dt_model.transform(tfidfData)
evaluate_model(dt_predictions)

# Random Forest Model
rf = RandomForestClassifier(labelCol="class_label", featuresCol="tfidfFeatures", numTrees=25)
rf_pipeline = Pipeline(stages=[rf])
rf_model = rf_pipeline.fit(tfidfData)
rf_predictions = rf_model.transform(tfidfData)
evaluate_model(rf_predictions)
