import pandas as pd
import matplotlib.pyplot as plt

# Parametri NP da considerare
np_values = [1, 2, 4, 8]

# Creazione del grafico
plt.figure(figsize=(10, 6))

# Caricamento e grafico dei dati per ogni valore di NP
for np_value in np_values:
    file_name = f"results/result_NP={np_value}.csv"
    
    # Carica i dati dal file CSV e plotta il grafico
    data = pd.read_csv(file_name, header=None, names=["SIZE", "Tempo"])
    plt.plot(data['SIZE'], data['Tempo'], label=f"NP={np_value}")

plt.xlabel('SIZE')
plt.ylabel('Tempo')
plt.title('Tempo di esecuzione per vari NP e SIZE')
plt.legend()
plt.savefig("results/plot.png")