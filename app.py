from pyspark.sql import SparkSession
from pyspark.sql.functions import col, translate, avg, stddev
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("Stock Analysis").getOrCreate()

data = spark.read.option("header","true").option("inferSchema","true").csv("./SP500_data.csv")

data.show()

data.printSchema()

data = data.withColumn("Open", translate(col("Open"), ",", "").cast("double"))
data = data.withColumn("Close", translate(col("Close"), ",", "").cast("double"))
data = data.withColumn("High", translate(col("High"), ",", "").cast("double"))
data = data.withColumn("Low", translate(col("Low"), ",", "").cast("double"))
data = data.withColumn("Volume", translate(col("Volume"), ",", "").cast("double"))

priceChange = data.withColumn("Price Change", col("Close") - col("Open"))
priceChange.select("Date", "Price Change").show()

pandas_df = priceChange.select("Date", "Price Change").toPandas()

pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])

pandas_df.set_index('Date', inplace=True)

pandas_df['Price Change'].plot(kind='line')
plt.title('Price Change Over Time')
plt.ylabel('Price Change')
plt.show()


dailyReturn = data.withColumn("Daily Return", (col("Close") - col("Open")) / col("Open"))
averageDailyReturn = dailyReturn.agg(avg("Daily Return"))
averageDailyReturn.show()

pandas_df = dailyReturn.select("Date", "Daily Return").toPandas()

pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])

pandas_df.set_index('Date', inplace=True)

pandas_df['Daily Return'].plot(kind='line')
plt.title('Daily Return Over Time')
plt.ylabel('Daily Return')
plt.show()


windowSpec = Window.orderBy("Date").rowsBetween(-2, 2)
movingAverage = data.withColumn("Moving Average", avg(col("Close")).over(windowSpec))
movingAverage.select("Date", "Moving Average").show()

pandas_df = movingAverage.select("Date", "Moving Average").toPandas()

pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])

pandas_df.set_index('Date', inplace=True)

pandas_df['Moving Average'].plot(kind='line')
plt.title('Moving Average Over Time')
plt.ylabel('Moving Average')
plt.show()

risk = dailyReturn.agg(stddev("Daily Return"))
risk.show()

pandas_df = dailyReturn.select("Daily Return").toPandas()

pandas_df['Daily Return'].plot(kind='hist', bins=30)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.show()


assembler = VectorAssembler(inputCols=["Open", "High", "Low", "Volume"], outputCol="features")
output = assembler.transform(data)

lr = LinearRegression(labelCol="Close", featuresCol="features")
model = lr.fit(output)

predictions = model.transform(output)
predictions.select("prediction", "Close").show()


pandas_df = predictions.select("Close", "prediction").toPandas()

# plot actual vs predicted prices
plt.scatter(pandas_df['Close'], pandas_df['prediction'])
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()


# if the model's predictions are perfect, the points would all lie on a straight line; the closer the points are to a straight line, the better the model's predictions





