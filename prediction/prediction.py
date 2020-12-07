import quinn
import requests

from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local')
spark = SparkSession(sc)

url = 'http://web.njit.edu/~jed34/TestDataset.csv'
r = requests.get(url, allow_redirects=True)
open('TestDataset.csv', 'wb').write(r.content)
test_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('TestDataset.csv')

def remove_quotations(s):
    return s.replace('"', "")

test_df = quinn.with_columns_renamed(remove_quotations)(test_df)
print(test_df.toPandas().head())