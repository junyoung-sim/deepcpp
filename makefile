COM=g++
VER=-std=c++2a
output: main.o mlp.o
	$(COM) $(VER) main.o mlp.o -o exec
	rm *.o
main.o: ./src/main.cpp
	$(COM) $(VER) -c ./src/main.cpp
mlp.o: ./src/mlp.cpp
	$(COM) $(VER) -c ./src/mlp.cpp
