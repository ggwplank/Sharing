EXEC = better_matrix_vector
SRC_DIR = src
BIN_DIR = bin
RESULTS_DIR = results
SCRIPT_DIR = scripts

# Variabili per file sorgente e file output
SRC = $(SRC_DIR)/$(EXEC).c
OUT = $(BIN_DIR)/$(EXEC)

# Parametri di esecuzione
NP = 3	# Numero di processi
SIZE = 4 # Dimensione della matrice#
FILE = $(RESULTS_DIR)/results.csv # File di output

all: $(OUT)

# Compilazione
$(OUT): $(SRC)
	@echo "Compilazione di $(SRC)..."
	@mkdir -p $(BIN_DIR) $(RESULTS_DIR)  # Crea le directory bin e results se non esistono
	mpicc -o $(OUT) $(SRC)

# Esecuzione con parametri di default
run: $(OUT)
	@echo "Esecuzione di $(OUT) con $(NP) processi e matrice di dimensione $(SIZE)x$(SIZE)..."
	mpirun -np $(NP) $(OUT) $(SIZE) $(FILE)

# Esecuzione dei test multipli
test:
	@echo "Esecuzione del test_runner..."
	python3 $(SCRIPTS_DIR)/run_tests.py

# Generazione dei grafici con lo script Python
plot:
	@echo "Generazione dei grafici con lo script Python..."
	python3 $(SCRIPT_DIR)/plot_results.py

# Pulizia dei file
clean:
	@echo "Pulizia dei file compilati..."
	rm -f $(OUT)

# Pulizia dei file di output
clean_results:
	@echo "Pulizia dei risultati..."
	rm -f $(RESULTS_DIR)/*.csv