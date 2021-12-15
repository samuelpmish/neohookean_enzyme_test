#!/bin/bash

# on M1 mac laptop: this compiles, but the test segfaults
$CXX main.cpp -g -std=c++17 -arch arm64 -Xclang -load -Xclang /opt/homebrew/Cellar/enzyme/0.0.25/lib/ClangEnzyme-13.dylib -O2 -fno-vectorize -fno-unroll-loops -fno-experimental-new-pass-manager -I../benchmark/include -L../benchmark/build/src/ -lbenchmark -lpthread 

# on M1 mac laptop: this compiles and runs without issue
#$CXX main.cpp -g -std=c++17 -arch arm64 -Xclang -load -Xclang /opt/homebrew/Cellar/enzyme/0.0.25/lib/ClangEnzyme-13.dylib -O2 -fno-experimental-new-pass-manager -I../benchmark/include -L../benchmark/build/src/ -lbenchmark -lpthread 
