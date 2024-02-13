# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, translate, avg, stddev
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create a SparkSession
spark = SparkSession.builder.appName("Stock Analysis").getOrCreate()

# Load the data from a CSV file
data = spark.read.option("header","true").option("inferSchema","true").csv("./SP500_data.csv")

# Display the first few rows of the DataFrame
data.show()

# Print the schema of the DataFrame
data.printSchema()

# Remove commas from the numeric columns and cast them to double
data = data.withColumn("Open", translate(col("Open"), ",", "").cast("double"))
data = data.withColumn("Close", translate(col("Close"), ",", "").cast("double"))
data = data.withColumn("High", translate(col("High"), ",", "").cast("double"))
data = data.withColumn("Low", translate(col("Low"), ",", "").cast("double"))
data = data.withColumn("Volume", translate(col("Volume"), ",", "").cast("double"))

# Calculate the change in price of the stock over time
priceChange = data.withColumn("Price Change", col("Close") - col("Open"))
priceChange.select("Date", "Price Change").show()

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
pandas_df = priceChange.select("Date", "Price Change").toPandas()

# Convert 'Date' to datetime
pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])

# Set 'Date' as the index of the DataFrame
pandas_df.set_index('Date', inplace=True)

# Plot the change in price over time
pandas_df['Price Change'].plot(kind='line')
plt.title('Price Change Over Time')
plt.ylabel('Price Change')
plt.show()

# This line graph shows the change in price of the stock over time. 
# If the line is going up, it means the price is increasing. If it's going down, the price is decreasing.

# Calculate the daily return of the stock
dailyReturn = data.withColumn("Daily Return", (col("Close") - col("Open")) / col("Open"))
averageDailyReturn = dailyReturn.agg(avg("Daily Return"))
averageDailyReturn.show()

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
pandas_df = dailyReturn.select("Date", "Daily Return").toPandas()

# Convert 'Date' to datetime
pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])

# Set 'Date' as the index of the DataFrame
pandas_df.set_index('Date', inplace=True)

# Plot the daily return over time
pandas_df['Daily Return'].plot(kind='line')
plt.title('Daily Return Over Time')
plt.ylabel('Daily Return')
plt.show()

# This line graph shows the daily return of the stock over time. 
# A positive number means the stock price tends to increase during the day, while a negative number means it tends to decrease.

# Calculate the moving average of the stock price
windowSpec = Window.orderBy("Date").rowsBetween(-2, 2)
movingAverage = data.withColumn("Moving Average", avg(col("Close")).over(windowSpec))
movingAverage.select("Date", "Moving Average").show()

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
pandas_df = movingAverage.select("Date", "Moving Average").toPandas()

# Convert 'Date' to datetime
pandas_df['Date'] = pd.to_datetime(pandas_df['Date'])

# Set 'Date' as the index of the DataFrame
pandas_df.set_index('Date', inplace=True)

# Plot the moving average over time
pandas_df['Moving Average'].plot(kind='line')
plt.title('Moving Average Over Time')
plt.ylabel('Moving Average')
plt.show()

# This line graph shows the moving average of the stock price over time. 
# It smooths out price fluctuations to help you identify the trend. 
# If the line is going up, the stock is in an uptrend. If it's going down, it's in a downtrend.

# Calculate the standard deviation of the daily returns, which represents the risk
risk = dailyReturn.agg(stddev("Daily Return"))
risk.show()

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
pandas_df = dailyReturn.select("Daily Return").toPandas()

# Plot a histogram of the daily returns
pandas_df['Daily Return'].plot(kind='hist', bins=30)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.show()

# This histogram shows the distribution of daily returns. The shape of the distribution gives you an idea about the risk and return of the stock. 
# If the distribution is wide (high standard deviation), the stock is more volatile and riskier. 
# If it's narrow (low standard deviation), the stock is less volatile and less risky.


# Create a feature vector by combining the input features
assembler = VectorAssembler(inputCols=["Open", "High", "Low", "Volume"], outputCol="features")
output = assembler.transform(data)

# Train a Linear Regression model on the data
lr = LinearRegression(labelCol="Close", featuresCol="features")
model = lr.fit(output)

# Make predictions using the model
predictions = model.transform(output)
predictions.select("prediction", "Close").show()

# This part of the code uses a Linear Regression model to predict future stock behavior based on the "Open", "High", "Low", and "Volume" features. 
# The model is trained on the data, and then it's used to make predictions. 
# The predictions are compared with the actual closing prices

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
pandas_df = predictions.select("Close", "prediction").toPandas()

# Plot actual vs predicted prices
plt.scatter(pandas_df['Close'], pandas_df['prediction'])
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

# This scatter plot shows the actual closing prices vs. the predicted prices. 
# If the model's predictions are perfect, the points would all lie on a straight line. 
# The closer the points are to a straight line, the better the model's predictions.





