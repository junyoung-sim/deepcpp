COM=g++
VER=-std=c++2a

output: classifier.o logreg.o reg.o math.o mlp.o
	$(COM) $(VER) classifier.o math.o mlp.o -o classifier
	$(COM) $(VER) logreg.o math.o mlp.o -o logreg
	$(COM) $(VER) reg.o math.o mlp.o -o reg
	rm *.o

classifier.o: classifier.cpp
	$(COM) $(VER) -c classifier.cpp
logreg.o: logreg.cpp
	$(COM) $(VER) -c logreg.cpp
reg.o: reg.cpp
	$(COM) $(VER) -c reg.cpp

math.o: ../src/math.cpp
	$(COM) $(VER) -c ../src/math.cpp
mlp.o: ../src/mlp.cpp
	$(COM) $(VER) -c ../src/mlp.cpp
