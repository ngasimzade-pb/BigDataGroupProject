import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import Row
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer, NGram
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark Session
sp = (SparkSession.builder
      .appName('Medical Document Classification')
      .config("spark.sql.crossJoin.enabled", "true")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate())

sp.sparkContext.setLogLevel('INFO')

# Load Data
print("\nLoading Air Quality Dataset...\n")
train_data = sp.read.option("delimiter", ";").csv('AirQualityUCI.csv', header=True, inferSchema=True)
test_data = sp.read.option("delimiter", ";").csv('AirQualityUCI.csv', header=True, inferSchema=True)


# Assuming 'AH' as a placeholder for the text column in Air Quality dataset
train_spark_df = train_data.withColumnRenamed("AH", "text").withColumnRenamed("class", "class_label")
test_spark_df = test_data.withColumnRenamed("AH", "text")



# Preprocessing
regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

words_data = stopwords_remover.transform(regex_tokenizer.transform(train_spark_df))


# TF-IDF
hash_tf = HashingTF(inputCol="filtered_tokens", outputCol="tfFeatures")
idf = IDF(inputCol="tfFeatures", outputCol="tfidfFeatures")
idf_model = idf.fit(hash_tf.transform(words_data))
tfidf_data = idf_model.transform(hash_tf.transform(words_data))

# Bag of Words (BOW)
count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="BOWFeatures")
bow_data = count_vectorizer.fit(words_data).transform(words_data)

print("\nhereeee3\n")

# N-Grams
n_value = 3
ngram = NGram(n=n_value, inputCol="filtered_tokens", outputCol="ngram_features")
ngram_data = ngram.transform(words_data)

# Evaluation Function
def evaluate_model(predictions, label_col="class_label", prediction_col="prediction"):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
    metrics = {
        "Accuracy": evaluator.setMetricName("accuracy").evaluate(predictions),
        "F1 Score": evaluator.setMetricName("f1").evaluate(predictions),
        "Precision": evaluator.setMetricName("weightedPrecision").evaluate(predictions),
        "Recall": evaluator.setMetricName("weightedRecall").evaluate(predictions),
    }
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Sub-methods for Evaluation
def evaluate_tfidf():
    train, test = tfidf_data.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="tfidfFeatures", labelCol="class_label", regParam=0.1)
    model = lr.fit(train)
    predictions = model.transform(test)
    print("--- TF-IDF Logistic Regression ---")
    evaluate_model(predictions)

    rf = RandomForestClassifier(featuresCol="tfidfFeatures", labelCol="class_label", numTrees=10)
    model = rf.fit(train)
    predictions = model.transform(test)
    print("--- TF-IDF Random Forest ---")
    evaluate_model(predictions)

def evaluate_bow():
    train, test = bow_data.randomSplit([0.8, 0.2], seed=42)
    dt = DecisionTreeClassifier(featuresCol="BOWFeatures", labelCol="class_label", maxDepth=10)
    model = dt.fit(train)
    predictions = model.transform(test)
    print("--- BOW Decision Tree ---")
    evaluate_model(predictions)

    rf = RandomForestClassifier(featuresCol="BOWFeatures", labelCol="class_label", numTrees=10)
    model = rf.fit(train)
    predictions = model.transform(test)
    print("--- BOW Random Forest ---")
    evaluate_model(predictions)

def evaluate_ngram():
    train, test = ngram_data.randomSplit([0.8, 0.2], seed=42)
    cv = CountVectorizer(inputCol="ngram_features", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="class_label", regParam=0.1)
    pipeline = Pipeline(stages=[cv, lr])
    model = pipeline.fit(train)
    predictions = model.transform(test)
    print("--- N-Gram Logistic Regression ---")
    evaluate_model(predictions)

    rf = RandomForestClassifier(featuresCol="features", labelCol="class_label", numTrees=10)
    pipeline = Pipeline(stages=[cv, rf])
    model = pipeline.fit(train)
    predictions = model.transform(test)
    print("--- N-Gram Random Forest ---")
    evaluate_model(predictions)


# Main Method
def main():
    print("Starting Evaluation for Medical Document Classification...")
    print("\nEvaluating TF-IDF Models:\n")
    evaluate_tfidf()

    print("\nEvaluating BOW Models:\n")
    evaluate_bow()

    print("\nEvaluating N-Gram Models:\n")
    evaluate_ngram()

if __name__ == "__main__":
    main()
