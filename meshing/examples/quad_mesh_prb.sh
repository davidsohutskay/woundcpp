#!/bin/bash
#
g++ -c -I/$HOME/include quad_mesh_prb.cpp
if [ $? -ne 0 ]; then
  echo "Errors compiling quad_mesh_prb.cpp"
  exit
fi
#
g++ quad_mesh_prb.o /$HOME/libcpp/$ARCH/quad_mesh.o -lm
if [ $? -ne 0 ]; then
  echo "Errors linking and loading quad_mesh_prb.o."
  exit
fi
#
rm quad_mesh_prb.o
#
mv a.out quad_mesh_prb
./quad_mesh_prb > quad_mesh_prb_output.txt
if [ $? -ne 0 ]; then
  echo "Errors running quad_mesh_prb."
  exit
fi
rm quad_mesh_prb
#
echo "Program output written to quad_mesh_prb_output.txt"
