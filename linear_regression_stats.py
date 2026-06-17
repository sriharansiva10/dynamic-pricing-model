import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#readind the data from the csv file
csv=pd.read_csv('ebike_dynamic_pricing_dataset_without_weather.csv')

# plot for first ten rows
head_csv=csv.head(5)

# linear regression plotting
slope,intercept,r,p,stderr=stats.linregress(head_csv['fuel_price'],head_csv['rental_price']) #plotting between fuel price and rental price

#finding f(x)
def func(x):
    return slope*x+intercept

#iterate over a rental price

linear_regression=list(map(func,head_csv['fuel_price']))

#plotting
#stop the plotting if the correlation is less than 0.6 or greater than -0.6
if abs(r) > 0.6 :
    plt.scatter(head_csv['fuel_price'],head_csv['rental_price'],color='#1010e3')
    plt.plot(head_csv['fuel_price'],linear_regression,color='#13e80c')
    plt.subplot(1,2,1)
else :
    print("no correlation between fuel price and rental price")

slope,intercept,r,p,stderr=stats.linregress(head_csv['fuel_price'],head_csv['demand_score'])

linear_regression=list(map(func,head_csv['fuel_price']))

if abs(r) > 0.6 :
    plt.scatter(head_csv['fuel_price'],head_csv['demand_score'])
    plt.plot(head_csv['fuel_price'],linear_regression,color='#13e80c')
    plt.subplot(1,2,2)
    plt.show()