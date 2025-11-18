cd ../build
rm fdtd
make -j6
clear
make -j6
cd ../test
../build/fdtd control_file.json