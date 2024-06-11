import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("social media prediction.csv")

#df[['period_start','period_end','followers_gained','followers_lost','followers_net','followers_total','subscribers_gained','subscribers_lost','subscribers_net','subscribers_total','views']] = df['period_start,period_end,followers_gained,followers_lost,followers_net,followers_total,subscribers_gained,subscribers_lost,subscribers_net,subscribers_total,views'].str.split(',', expand=True)
#These 2 lines split 1st column of the csv file into a number of columns

#df.drop(['period_start,period_end,followers_gained,followers_lost,followers_net,followers_total,subscribers_gained,subscribers_lost,subscribers_net,subscribers_total,views'], axis=1, inplace=True)
#This line is used to drop the 1st column

#df.to_csv('social media prediction.csv', index=False)
#This line is used to store the new columns in the csv file

df.drop(df.tail(1).index, inplace=True)
print(df.head(5))

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Number of Followers I Gained Every Month")
sns.barplot(x="followers_gained", y="period_end", data=df)
plt.show()

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Followers At The End of Every Month")
sns.barplot(x="followers_total", y="period_end", data=df)
plt.show()

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Views Every Month")
sns.barplot(x="views", y="period_end", data=df)
plt.show()

from autots import AutoTS
model = AutoTS(forecast_length=4, frequency='infer', ensemble='simple')
model = model.fit(df, date_col='period_end', value_col='followers_gained', id_col=None)
prediction = model.predict(5)
forecast = prediction.forecast
print(forecast)

