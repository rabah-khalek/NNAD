# Check this http://mrbook.org/blog/tutorials/make/
# and this https://gist.github.com/ghl3/3975167

INCLUDES = -I$(PWD)/inc -I/usr/local/include/eigen3 $(shell root-config --cflags)
CXX = clang++ $(INCLUDES)
CXXFLAGS = -std=c++11

# Assumes that the script is in MyScript.C
# and it must contain a main() function
all: main

main: 

% : %.cc #for each target X, if X.c exists and is newer than X (or X doesn't exist), run the command below
	clang++ -O3 -g -Wall -stdlib=libc++ -std=c++11 -L/usr/local/lib/ -lceres -lglog $(shell root-config --libs) $(INCLUDES) $? -o FFNN

#	chmod +x nNNPDF
	rm -r FFNN.dSYM
	@echo "======= make is done ======="
clean:
	rm src/*o FFNN
