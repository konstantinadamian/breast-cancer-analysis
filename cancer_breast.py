import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

my_data = pd.read_csv("C:\\Users\\Konny\\Downloads\\archive (2)\\breast-cancer.csv")
data = pd.DataFrame(my_data)
print(data.isnull().sum())
#statistics
print(data.describe())

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
print(data.columns)
grouped = data.groupby('diagnosis').mean()
print(grouped)

# radius_mean > 15 and diagnosis == 1
filtered_data = data[(data['radius_mean'] > 15) & (data['diagnosis'] == 1)]
print(filtered_data)
# scatter plot
x = data['radius_mean']
y = data['area_mean']
plt.scatter(x, y, alpha=0.9,marker='.',color='r',linewidths=0.1,edgecolors='b')
plt.xlabel('Radius Mean')
plt.ylabel('Area Mean')
plt.title('Scatter Plot of Radius Mean vs Area Mean')
plt.grid(True)
plt.show()

