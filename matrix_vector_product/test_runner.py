import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Parametri
np_values = [1, 2, 4, 8]  # Lista di valori di NP
max_size = 50          # Dimensione massima
size_values = list(range(0, 1000, 10)) + list(range(1000, 4000, 100)) + list(range(4000, 30000, 1000))


# Funzione per eseguire il comando con parametri NP e SIZE specifici
def run_make(np_value, size):
    file_name = f"result_NP={np_value}.csv"
    command = f"make run NP={np_value} SIZE={size} FILE={file_name}"
    subprocess.run(command, shell=True, check=True)
    return file_name

# Creazione dei file CSV
for np_value in np_values:
    for size in size_values:
        run_make(np_value, size)

# Creazione del grafico
plt.figure(figsize=(10, 6))

for np_value in np_values:
    file_name = f"result_NP={np_value}.csv"
    
    # Carica il CSV una sola volta per ogni NP
    data = pd.read_csv(file_name, header=None, names=["SIZE", "Tempo"])
    
    # Aggiungi la curva al grafico
    plt.plot(data['SIZE'], data['Tempo'], label=f"NP={np_value}")

# Configura il grafico
plt.xlabel('SIZE')
plt.ylabel('Tempo')
plt.title('Tempo di esecuzione per vari NP e SIZE')
plt.legend()
plt.show()
