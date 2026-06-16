import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

file=pd.read_csv('ebike_dynamic_pricing_dataset_without_weather.csv')

slope,intercept,r,p,stderr=stats.linregress(file['fuel_price'],file['demand_score'])
def func(x):
    return slope*x+intercept
model=list(map(func,file['fuel_price']))
plt.scatter(file['fuel_price'],file['demand_score'])
plt.plot(file['fuel_price'],model)
plt.show()
print(r)