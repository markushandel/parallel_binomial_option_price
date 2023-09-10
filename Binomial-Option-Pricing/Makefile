CXX = /usr/local/cuda-11.4/bin/nvcc
CXXFLAGS = -std=c++11

all: main

main: main.cu Options/Options/option.cpp
	$(CXX) $(CXXFLAGS) -IOptions/Options $< Options/Options/option.cpp -o $@

clean:
	rm -f main