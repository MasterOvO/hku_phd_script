import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import json
import numpy as np
from matplotlib import rcParams
import pandas as pd
import matplotlib.cm as cm


rcParams.update({'figure.autolayout': True})

#arr: datapoint with startpoint, endpoint, pupa rate
data = pd.read_excel(r"raw\2025-9-10 phm topto egfr timepoint small n dissect\combine\combine 15-8 22-8 3-9 10-9.xlsx")
data = data.sort_values("light duration ")

my_cmap = plt.get_cmap('inferno')
count = 0
duration1arr= []
duration2arr= []
for ind in data.index:
    if len(data["light duration "].loc[ind].split("-"))<2:
        duration1arr.append(0)
        duration2arr.append(144)
        continue
    duration1, duration2 = data["light duration "].loc[ind].split("-")
    
    duration1arr.append(int(duration1))
    duration2arr.append(int(duration2))

data["duration1"] = duration1arr
data["duration2"] = duration2arr

data = data.sort_values(["duration1","duration2"])

print(data)

# 1. Explicitly initialize a single figure and axes object
fig, ax = plt.subplots()

count = 0
for ind in data.index:
    count += 1
    # Use ax.barh instead of plt.barh to target the specific axes
    ax.barh(
        count, 
        data["duration2"].loc[ind] - data["duration1"].loc[ind], 
        left=data["duration1"].loc[ind], 
        color=my_cmap(data["%"].loc[ind])
    )

norm = plt.Normalize(0, 1)
ax.set_xlim(0, 144)

# Ensure text vectors remain fully editable as fonts in Adobe Illustrator
plt.rcParams['svg.fonttype'] = 'none'

# 2. Instantiate ScalarMappable
sm = cm.ScalarMappable(cmap=my_cmap, norm=norm)

# 3. Attach colorbar to the explicit figure and steal space from the exact 'ax'
fig.colorbar(sm, ax=ax, label="pupa percentage")

# 4. Apply structural labels specifically to the active axes object
ax.set_xlabel("hours after egg laying \n (light duration)")
ax.set_yticks([])
ax.set_ylabel("experiment")

# 5. Save and display the unified graphic layout
plt.savefig("timepoint combine.svg", format="svg", bbox_inches="tight")
plt.show()