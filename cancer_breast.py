import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

my_data = pd.read_csv("C:\\Users\\Konny\\Downloads\\archive (2)\\breast-cancer.csv")
data = pd.DataFrame(my_data)
print(data.isnull().sum())
#statistics
print(data.describe())

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
print(data.columns)
grouped = data.groupby('diagnosis').mean()
print(grouped)
# radius_mean > 15 and diagnosis == 
filtered_data = data[(data['radius_mean'] > 15) & (data['diagnosis'] == 1)]
print(filtered_data)
# scatter plot
x = data['radius_mean']
y = data['area_mean']

spearmanr_corr, _ = spearmanr(x,y)
print(f"Spearmanr Correlation: {spearmanr_corr}")

plt.scatter(x, y, alpha=0.9,marker='.',color='r',linewidths=0.1,edgecolors='b')
plt.xlabel('Radius Mean')
plt.ylabel('Area Mean')
plt.title('Scatter Plot of Radius Mean vs Area Mean')
plt.grid(True)
plt.show()

model = LinearRegression()

reshape_x = np.array(x).reshape(-1, 1)
model.fit(reshape_x, y)
y_pred = model.predict(reshape_x)

print(y_pred)

plt.scatter(reshape_x, y, color='lightcyan',edgecolors='navy', label="Data")
plt.plot(reshape_x, y_pred, color='r', label="Linear Fit")
plt.xlabel("Radius Mean")
plt.ylabel("Area Mean")
plt.legend()
plt.grid(True, color='lavender')
plt.show()



