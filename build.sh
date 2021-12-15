#!/bin/bash

# on M1 mac laptop: this compiles and runs without issue
#$CXX main.cpp -g -std=c++17 -arch arm64 -Xclang -load -Xclang /opt/homebrew/Cellar/enzyme/0.0.25/lib/ClangEnzyme-13.dylib -O2 -fno-experimental-new-pass-manager -I../benchmark/include -L../benchmark/build/src/ -lbenchmark -lpthread 

# on M1 mac laptop: this compiles with `-fno-vectorize -fno-unroll-loops` enabled, but the test segfaults
#$CXX main.cpp -g -std=c++17 -arch arm64 -Xclang -load -Xclang /opt/homebrew/Cellar/enzyme/0.0.25/lib/ClangEnzyme-13.dylib -O2 -fno-vectorize -fno-unroll-loops -fno-experimental-new-pass-manager -I../benchmark/include -L../benchmark/build/src/ -lbenchmark -lpthread 

# on Ubuntu 20.04: this compiles and runs without issue
/usr/bin/clang++-12 main.cpp -g -std=c++17 -Xclang -load -Xclang /home/sam/code/Enzyme/enzyme/build/Enzyme/ClangEnzyme-12.so -O3 -I../benchmark/include -L../benchmark/build/src -lbenchmark -lpthread

# on Ubuntu 20.04: this compiles with `-fno-vectorize -fno-unroll-loops` enabled and runs without issue
#/usr/bin/clang++-12 main.cpp -g -std=c++17 -Xclang -load -Xclang /home/sam/code/Enzyme/enzyme/build/Enzyme/ClangEnzyme-12.so -O3 -fno-vectorize -fno-unroll-loops -I../benchmark/include -L../benchmark/build/src -lbenchmark -lpthread
