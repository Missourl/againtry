import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

data=pd.read_csv("resutohnepowermeanobsreward.csv", delimiter=',' )
y=data.iloc[:,1]

window_size = 5

i = 0
# Initialize an empty list to store moving averages
moving_averages = []

# Loop through the array to consider
# every window of size 3
while i < len(y) - window_size + 1:
    # Store elements from i to i+window_size
    # in list to get the current window
    window = y[i: i + window_size]

    # Calculate the average of current window
    window_average = round(sum(window) / window_size, 2)

    # Store the average of current
    # window in moving average list
    moving_averages.append(window_average)

    # Shift window to right by one position
    i += 1

X=list(range(1,1000))
# Plotting both the curves simultaneously
plt.plot(X[0:996], y[0:996], color='g', label='episodic reward')
plt.plot(X[0:996], moving_averages[0:996], color='b', label='moving average')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()