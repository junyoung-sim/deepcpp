COM=g++
VER=-std=c++2a

output: reg.o logreg.o classifier.o math.o mlp.o
	$(COM) $(VER) reg.o math.o mlp.o -o reg
	$(COM) $(VER) logreg.o math.o mlp.o -o logreg
	$(COM) $(VER) classifier.o math.o mlp.o -o classifier
	rm *.o

reg.o: reg.cpp
	$(COM) $(VER) -c reg.cpp
logreg.o: logreg.cpp
	$(COM) $(VER) -c logreg.cpp
classifier.o: classifier.cpp
	$(COM) $(VER) -c classifier.cpp

math.o: ../src/math.cpp
	$(COM) $(VER) -c ../src/math.cpp
mlp.o: ../src/mlp.cpp
	$(COM) $(VER) -c ../src/mlp.cpp
