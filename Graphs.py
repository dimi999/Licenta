import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter
import os

# Given data
data = {
    "grok_asg_anomaly": {"AMMMO": 1.36, "Gorilla": 1.25},
    "occupancy_t4013": {"AMMMO": 1.09, "Gorilla": 1.06},
    "Sunspots": {"AMMMO": 1.33, "Gorilla": 1.17},
    "monthly-beer-production": {"AMMMO": 1.35, "Gorilla": 1.26},
    "monthly-housing": {"AMMMO": 1.18, "Gorilla": 1.17},
    "cpu_utilization": {"AMMMO": 1.90, "Gorilla": 1.61},
    "art-price": {"AMMMO": 1.01, "Gorilla": 1.10},
    "Gold": {"AMMMO": 1.81, "Gorilla": 1.70},
    "Electric_Production": {"AMMMO": 1.00, "Gorilla": 1.08},
    "daily-temperatures": {"AMMMO": 1.36, "Gorilla": 1.17},
    "oil": {"AMMMO": 1.11, "Gorilla": 1.12},
    "rogue_agent_key_updown": {"AMMMO": 6.32, "Gorilla": 5.94}
}

# Extract the filenames, timestamps, and metrics
filenames = list(data.keys())
AMMMO = [data[file]["AMMMO"] for file in filenames]
Gorilla = [data[file]["Gorilla"] for file in filenames]

# Create the plot
plt.figure(figsize=(14, 8))

# Plot timestamps
plt.plot(filenames, AMMMO, marker='o', linestyle='-', color='b', label='AMMMO', )

# Plot metrics
plt.plot(filenames, Gorilla, marker='o', linestyle='-', color='r', label='Gorilla')


# Add base axis line at 1
plt.axhline(y=1, color='gray', linestyle='--')

# Add titles and labels
plt.title('Rata de compresie valori metrice')
plt.xlabel('Serii de timp')
plt.ylabel('Rata de compresie')
plt.xticks(rotation=45, ha='right')
plt.legend()
#plt.yscale('log', base=2)
#plt.gca().yaxis.set_major_locator(LogLocator(base=2))
#plt.gca().yaxis.set_major_formatter(LogFormatter(base=2))
plt.grid(True)

# Show the plot
plt.tight_layout()
my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/DCA_metric.pdf', format='pdf')
plt.show()

#%%
import os
# Given data
data = {
    "rogue_agent_key_updown": {"My-AMMMO": 9.19, "Lazy-AMMMO": 11.27, "Gorilla": 10.62},
    "oil": {"My-AMMMO": 1.64, "Lazy-AMMMO": 1.60, "Gorilla": 1.51},
    "grok_asg_anomaly": {"My-AMMMO": 2.74, "Lazy-AMMMO": 2.67, "Gorilla": 2.46},
    "monthly-housing": {"My-AMMMO": 2.00, "Lazy-AMMMO": 1.96, "Gorilla": 1.91},
    "art-price": {"My-AMMMO": 2.00, "Lazy-AMMMO": 2.00, "Gorilla": 2.17},
    "cpu_utilization": {"My-AMMMO": 3.75, "Lazy-AMMMO": 3.69, "Gorilla": 3.14},
}


# Extract the filenames, timestamps, and metrics
labels = list(data.keys())
my_ammo = [data[key]["My-AMMMO"] for key in labels]
lazy_ammo = [data[key]["Lazy-AMMMO"] for key in labels]
gorilla = [data[key]["Gorilla"] for key in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

rects1 = ax.bar(x - width, my_ammo, width, label='My-AMMMO')
rects2 = ax.bar(x, lazy_ammo, width, label='Lazy-AMMMO')
rects3 = ax.bar(x + width, gorilla, width, label='Gorilla')

ax.set_xlabel('Seriile de timp')
ax.set_ylabel('Valori')
ax.set_title('Rata de compresie pe valori metrice')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

fig.tight_layout()

my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/compresia_totala.pdf', format='pdf')

#%%
data = {
    "grok_asg_anomaly": {"AMMMO": 2.67, "Gorilla": 2.46},
    "occupancy_t4013": {"AMMMO": 1.63, "Gorilla": 1.5},
    "Sunspots": {"AMMMO": 1.37, "Gorilla": 1.14},
    "monthly-beer-production": {"AMMMO": 1.38, "Gorilla": 1.18},
    "monthly-housing": {"AMMMO": 1.96, "Gorilla": 1.91},
    "cpu_utilization": {"AMMMO": 3.69, "Gorilla": 3.14},
    "art-price": {"AMMMO": 2.0, "Gorilla": 2.17},
    "Gold": {"AMMMO": 2.19, "Gorilla": 1.93},
    "Electric_Production": {"AMMMO": 1.72, "Gorilla": 1.79},
    "daily-temperatures": {"AMMMO": 2.65, "Gorilla": 2.31},
    "oil": {"AMMMO": 1.68, "Gorilla": 1.51},
    "rogue_agent_key_updown": {"AMMMO": 11.27, "Gorilla": 10.62}
}

# Calculate percentage increases
percentage_increases = {key: ((value["AMMMO"] - value["Gorilla"]) / value["Gorilla"]) * 100 for key, value in data.items()}

# Extract keys and values for plotting
labels = list(percentage_increases.keys())
values = list(percentage_increases.values())

# Create the bar graph
plt.figure(figsize=(14, 8))
bars = plt.bar(labels, values, color='skyblue')

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

# Add titles and labels
plt.title('Creșterea procentuală a ratei de compresie')
plt.xlabel('Serii de timp')
plt.ylabel('Creșterea procentului (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')

# Show the plot
plt.tight_layout()
my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/DCA_percentage.pdf', format='pdf')
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    "Dataset": ["grok_asg_anomaly.csv", "occupancy_t4013.csv", "Sunspots.csv", "monthly-beer-production.csv", 
                "monthly-housing.csv", "cpu_utilization.csv", "art-price.csv", "Gold.csv", 
                "Electric_Production.csv", "daily-temperatures.csv", "oil.csv", "rogue_agent_key_updown.csv"],
    "Huffman_Valori_Timestamps": [45.6395061728395, 6.720430107526882, 16.305961754780654, 7.600798403193613, 
                                  5.267813267813268, 43.76662143826323, 43.76662143826323, 23.870410367170628, 
                                  7.420560747663552, 42.31884057971015, 20.427672955974842, 32.383853769992385],
    "Huffman_Valori_Metrics": [0.895130632702971, 0.49957536094319827, 0.3695979194819102, 0.20015768725361366, 
                               0.22398662766402005, 8.047904191616766, 0.21000957074866694, 1.02248126561199, 
                               0.16757241597636258, 1.3167388167388168, 0.25384915982805784, 1.1023540391994193],
    "Global_Huffman_Timestamps": [7.968958827333477, 4.065040650406504, 3.3408619497580085, 3.311304347826087, 
                                  4.357723577235772, 7.962478400394964, 7.964444444444444, 4.949395432154053, 
                                  4.435754189944134, 7.954235903023699, 5.117647058823529, 7.897473997028232],
    "Global_Huffman_Metrics": [1.1248782862706914, 1.1247328759419637, 1.5528655597214782, 1.5755068266446008, 
                               1.199776161163962, 1.4043887147335423, 0.9583741866476513, 1.8519542541158729, 
                               0.9517530716212167, 1.5841154451255899, 1.1608291636883488, 2.681803847366761],
    "Local_Huffman_Timestamps": [7.414360208584036, 3.15905860053704, 3.147882736156352, 2.2532544378698223, 
                                 2.135458167330677, 7.335910848305663, 7.335910848305663, 4.81044613710555, 
                                 2.7098976109215016, 6.787540678754068, 3.983646770237122, 7.087847974662444],
    "Local_Huffman_Metrics": [0.9305978602894902, 0.8338544923910778, 1.1440296740588747, 0.5633969522118657, 
                              0.34642106963968333, 1.4843994477680627, 0.8096792007630905, 1.7124264022311744, 
                              0.2588005215123859, 1.4454729963863175, 0.5953806672369547, 2.063777119836917]
}

df = pd.DataFrame(data)

# Plot for metric data
plt.figure(figsize=(14, 7))
plt.plot(df["Dataset"], df["Huffman_Valori_Metrics"], marker='o', label='Huffman Valori')
plt.plot(df["Dataset"], df["Global_Huffman_Metrics"], marker='o', label='Global Huffman')
plt.plot(df["Dataset"], df["Local_Huffman_Metrics"], marker='o', label='Local Huffman')
plt.title('Rata de compresie pentru valori metrice')
plt.xlabel('Serii de timp')
plt.ylabel('Rata de compresie')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/Huffman_metrici.pdf', format='pdf')
plt.show()

# Plot for temporal data
plt.figure(figsize=(14, 7))
plt.plot(df["Dataset"], df["Huffman_Valori_Timestamps"], marker='o', label='Huffman Valori')
plt.plot(df["Dataset"], df["Global_Huffman_Timestamps"], marker='o', label='Global Huffman')
plt.plot(df["Dataset"], df["Local_Huffman_Timestamps"], marker='o', label='Local Huffman')
plt.title('Rata de compresie pentru valori temporale')
plt.xlabel('Serii de timp')
plt.ylabel('Rata de compresie')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/Huffman_timp.pdf', format='pdf')
plt.show()

#%%
def f1(val):
    return -np.tanh(val - 1.2) / 0.5

def f2(val):
    return -np.tanh(val - 1) / 0.8

# Generate values for val
val = np.linspace(-2, 4, 400)

# Calculate the function values
y1 = f1(val)
y2 = f2(val)

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(val, y1, label='Formula 1')
plt.plot(val, y2, label='Formula 2')
plt.scatter([1, 1.2], [0, 0], c='red')
plt.xlabel('Ox')
plt.ylabel('Oy')
plt.title('Funcțiile de premiere folosite')
plt.legend()
plt.grid(True)
my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/funcii.pdf', format='pdf')
plt.show()
#%%
data = {
    "AIZ_stocks": {"My-AMMMO": 1.20, "Lazy-AMMMO": 1.15, "Gorilla": 1.17},
    "cpu_utilization_77": {"My-AMMMO": 1.44, "Lazy-AMMMO": 1.40, "Gorilla": 1.24},
    "nz_weather": {"My-AMMMO": 1.33, "Lazy-AMMMO": 1.23, "Gorilla": 1.10},
    "occupancy_6005": {"My-AMMMO": 1.14, "Lazy-AMMMO": 1.09, "Gorilla": 1.04},
    "load_balancer_spikes": {"My-AMMMO": 4.40, "Lazy-AMMMO": 5.40, "Gorilla": 5.43},
    "2014_apple_stock": {"My-AMMMO": 1.11, "Lazy-AMMMO": 1.09, "Gorilla": 1.15},
}

# Calculate percentage increases
percentage_increases = {"sunspots": 0.67,
                        "monthly-beer-production":3.69,
                        "ambient-temperature":-0.91,
                        "gold":8.06,
                        "2014_apple_stock":4.74,}

# Extract keys and values for plotting
labels = list(percentage_increases.keys())
values = list(percentage_increases.values())

# Create the bar graph
plt.figure(figsize=(14, 8))
bars = plt.bar(labels, values, color='skyblue')

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

# Add titles and labels
plt.title('Creșterea procentuală a ratei de compresie pe valori metrice')
plt.xlabel('Serii de timp')
plt.ylabel('Creșterea procentului (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')

# Show the plot
plt.tight_layout()
my_path = os.path.dirname(__file__)
plt.savefig(my_path + '/imagini/AMMMO-RL.pdf', format='pdf')
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
import os


my_path = os.path.dirname(__file__)


x = np.linspace(0, 10, 200)

def f1(x):
    return np.sin(2 * np.pi * 0.5 * x) + 0.4


def f2(x):
    return np.sin(2 * np.pi * 2 * x) + 0.4

y1 = f1(x)
y2 = f2(x)

plt.figure()
plt.plot(x, y1)
plt.title('Frecvența mică')
plt.xlabel('timpul')
plt.ylabel('amplitudinea / valoarea')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.savefig(my_path + '/imagini/frec_mic.pdf', format='pdf')
plt.figure()
plt.plot(x, y2)
plt.title('Frecvența mare')
plt.xlabel('timpul')
plt.ylabel('amplitudinea / valoarea')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.savefig(my_path + '/imagini/frec_mare.pdf', format='pdf')

#%%
import matplotlib.pyplot as plt

# Data for My-AMMMO
my_ammo_compression = [
    8.179145122129889, 8.072480765821066, 5.744674790482472, 5.877158008429622,
    8.568367592212265, 6.110882347300962, 5.980085188227895, 9.088528685407914,
    5.9593256982793115, 7.5585975321986245
]

my_ammo_decompression = [
    0.10355791614418536, 0.11265553158557746, 0.12441741959954034, 0.12570524927281904,
    0.0905100431141255, 0.12711620680996955, 0.13243607902626933, 0.08568284405740682,
    0.1299922922642798, 0.1119421617729924
]

# Data for Lazy-AMMMO
lazy_ammo_compression = [
    0.41283281518730625, 0.37163002056311506, 0.5486796059266494, 0.5765848648601436,
    0.5846683635930741, 0.3817463753113533, 0.5147451349955783, 0.5947338355766548,
    0.1322988482881276, 0.6059721112251282
]

lazy_ammo_decompression = [
    0.08341410263155778, 0.07934829427825504, 0.1237363195527954, 0.1275299093261551,
    0.12837017578509877, 0.07952464314341996, 0.11183967313235377, 0.13928989787678142,
    0.03069165775094407, 0.13157665729522705
]

# Data for Gorilla
gorilla_compression = [
    0.17254972619403786, 0.167243018279998, 0.1735666583728239, 0.19935414131732884,
    0.21643105238803978, 0.16826022760898976, 0.20905382636879088, 0.16510461991070935,
    0.206777078397045, 0.16878325300665087
]

gorilla_decompression = [
    0.09835686143466661, 0.10725762551720171, 0.11255157131680175, 0.07058937311956318,
    0.06697304514856298, 0.0927506028987048, 0.07201464375573711, 0.10341034231674996,
    0.08875100539854366, 0.10672409295725731
]

rl_ammo_compression = [
    0.799956606395209, 0.909042002549812, 0.512612954018608, 0.7385460298452805,
    0.5142644993842594, 0.5152153353842478, 0.7955173947917882, 0.5059038835858541,
    0.5221094877000839, 0.7750916836866691, 0.7288936358779223, 0.7224794644028394,
    0.5432768237023127, 0.5725463704457359
]

rl_ammo_decompression = [
    0.13737536188381821, 0.12884389108686306, 0.1337645309312003, 0.1360854106162911,
    0.1351363838665069, 0.13866890517492145, 0.13353219672815125, 0.13541489366501097,
    0.14101110753558932, 0.13175295360052763, 0.13560501497183272, 0.14208145995638263,
    0.13932763111023677, 0.14662033035641625
]

# Plotting the box plots for compression times including RL-AMMMO
plt.figure(figsize=(12, 6))
plt.boxplot([my_ammo_compression, lazy_ammo_compression, gorilla_compression, rl_ammo_compression], patch_artist=True)
plt.title('Timpul de compresie')
plt.ylabel('Timp (s)')
plt.xticks([1, 2, 3, 4], ['My-AMMMO', 'Lazy-AMMMO', 'Gorilla', 'RL-AMMMO'])
plt.grid(True)
plt.savefig(my_path + '/imagini/compresie_timp_tot.pdf', format='pdf')
plt.show()
plt.figure()
# Decompression times box plots
plt.boxplot([my_ammo_decompression, lazy_ammo_decompression, gorilla_decompression, rl_ammo_decompression], patch_artist=True)
plt.title('Timpul de decompresie')
plt.ylabel('Timp (s)')
plt.xticks([1, 2, 3, 4], ['My-AMMMO', 'Lazy-AMMMO', 'Gorilla', 'RL-AMMMO'])
plt.grid(True)
plt.savefig(my_path + '/imagini/decompresie_timp_tot.pdf', format='pdf')
plt.show()



