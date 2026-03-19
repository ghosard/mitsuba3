```
# create a new dir for build
mkdir build
cd build
# generate makefile [choose one]
cmake .. # for cmake only
cmake -G Ninja .. # for ninja
# build
cmake -j8 # build with 8 threads cmake
ninja -j8 # build with 8 threads ninja
```


For custom integrator, add it in ./src/integrators/your_integrator.cpp, and add its name in integrator folder's CMakeLists.txt like ```add_plugin(pathuH     pathuH.cpp) / add_plugin(pathuHN    pathuHN.cpp)```
