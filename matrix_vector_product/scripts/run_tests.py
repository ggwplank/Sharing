import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Parametri
np_values = [1, 2, 4, 8]  # Lista di valori di NP
size_values = list(range(0, 1000, 10)) + list(range(1000, 4000, 100)) + list(range(4000, 30000, 1000))


# Funzione per eseguire il comando con parametri NP e SIZE specifici
def run_make(np_value, size):
    file_name = f"results/result_NP={np_value}.csv"
    command = f"make run NP={np_value} SIZE={size} FILE={file_name}"
    subprocess.run(command, shell=True, check=True)
    return file_name

# Esecuzione del comando per ogni combinazione di NP e SIZE
for np_value in np_values:
    for size in size_values:
        run_make(np_value, size)
