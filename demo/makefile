COM=g++
VER=-std=c++2a

output: reg.o logreg.o classifier.o mnist.o math.o mlp.o
	$(COM) $(VER) reg.o math.o mlp.o -o ./reg/reg
	$(COM) $(VER) logreg.o math.o mlp.o -o ./logreg/logreg
	$(COM) $(VER) classifier.o math.o mlp.o -o ./classifier/classifier
	$(COM) $(VER) mnist.o math.o mlp.o -o ./mnist/mnist
	rm *.o

reg.o: ./reg/reg.cpp
	$(COM) $(VER) -c ./reg/reg.cpp
logreg.o: ./logreg/logreg.cpp
	$(COM) $(VER) -c ./logreg/logreg.cpp
classifier.o: ./classifier/classifier.cpp
	$(COM) $(VER) -c ./classifier/classifier.cpp

mnist.o: ./mnist/mnist.cpp
	$(COM) $(VER) -c ./mnist/mnist.cpp

math.o: ../src/math.cpp
	$(COM) $(VER) -c ../src/math.cpp
mlp.o: ../src/mlp.cpp
	$(COM) $(VER) -c ../src/mlp.cpp

clean:
	rm ./classifier/classifier
	rm ./logreg/logreg
	rm ./mnist/mnist
	rm ./reg/reg