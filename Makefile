all:
	g++ dct.cpp -o dct `pkg-config --libs opencv`
