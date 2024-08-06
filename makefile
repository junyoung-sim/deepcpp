COM=g++
VER=-std=c++2a
output: main.o math.o mlp.o cnn.o
	$(COM) $(VER) main.o math.o mlp.o cnn.o -o exec
	rm *.o
main.o: ./src/main.cpp
	$(COM) $(VER) -c ./src/main.cpp
math.o: ./src/math.cpp
	$(COM) $(VER) -c ./src/math.cpp
mlp.o: ./src/mlp.cpp
	$(COM) $(VER) -c ./src/mlp.cpp
cnn.o: ./src/cnn.cpp
	$(COM) $(VER) -c ./src/cnn.cpp
