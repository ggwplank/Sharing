import pandas as pd
import matplotlib.pyplot as plt

# Lista dei nomi dei file da processare
file_names = [
    "times i-j-k.csv",
    "times i-k-j.csv",
    "times j-i-k.csv",
    "times j-k-i.csv",
    "times k-i-j.csv",
    "times k-j-i.csv"
]

# Colori per le linee nel grafico combinato
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# Creazione della figura e degli assi per il grafico combinato
plt.figure(figsize=(10, 6))

# Itera su ogni file e aggiunge i dati al grafico combinato
for i, file_name in enumerate(file_names):
    # Leggi il file CSV
    df = pd.read_csv(file_name)

    # Estrai le colonne di interesse
    N = df['N']
    FLOPS = df['FLOPS']

    # Aggiungi al grafico combinato
    plt.plot(N, FLOPS, marker='o', color=colors[i], label=file_name.split('.')[0])

# Impostazioni per il grafico combinato
plt.xlabel('N')
plt.ylabel('FLOPS')
plt.title('Grafico combinato per tutte le permutazioni')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Mostra il grafico combinato
plt.show()
