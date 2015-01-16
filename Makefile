all: maincode clean

maincode: cudacode
	g++ -std=c++11 superpixels.cpp superpixels.o -o main `pkg-config --cflags --libs gtk+-2.0 opencv` -lGL -lGLU -lGLEW -lglut -lglut -lglfw -L/usr/local/cuda/lib64 -lcudart

cudacode:
	nvcc -c superpixels.cu -o superpixels.o -lGL -lGLU -lGLEW -lglut 

clean:
	rm -f *.o