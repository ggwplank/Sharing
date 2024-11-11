EXEC = better_matrix_vector
SRC = $(EXEC).c
NP = 3
SIZE = 4  # Dimensione di default della matrice
FILE = "results.csv"

all: $(EXEC)

$(EXEC): $(SRC)
	@echo "Compilazione di $(SRC)..."
	mpicc -o $(EXEC) $(SRC)

run: $(EXEC)
	@echo "Esecuzione di $(EXEC) con $(NP) processi e matrice di dimensione $(SIZE)x$(SIZE)..."
	mpirun -np $(NP) ./$(EXEC) $(SIZE) $(FILE)

clean:
	@echo "Pulizia dei file..."
	rm -f $(EXEC)
