import pandas as pd
import matplotlib.pyplot as plt

csv=pd.read_csv('ebike_dynamic_pricing_dataset_without_weather.csv')

print(pd.DataFrame(csv).corr())
