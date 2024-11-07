EXEC = better_matrix_vector
SRC = $(EXEC).c
NP = 3


all: $(EXEC)

$(EXEC): $(SRC)
	@echo "Compilazione di $(SRC)..."
	mpicc -o $(EXEC) $(SRC)

run: $(EXEC)
	@echo "Esecuzione di $(EXEC) con $(NP) processi..."
	mpirun -np $(NP) ./$(EXEC)