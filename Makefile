INCLUDES = -I$(PWD)/inc -I/usr/local/include/eigen3
CXX = clang++

# Assumes that the script is in MyScript.C
# and it must contain a main() function
all: main

main: 

% : %.cc #for each target X, if X.c exists and is newer than X (or X doesn't exist), run the command below
	$(CXX) -O3 -g -Wall -stdlib=libc++ -std=c++11 -L/usr/local/lib/ -lceres -lglog -lyaml-cpp $(INCLUDES) $? -o NNAGD

	rm -r NNAGD.dSYM
	@echo "======= make is done ======="
clean:
	rm src/*o NNAGD
