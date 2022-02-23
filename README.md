# Prerequires
- Deepstream-triton 6.0
# Run
## Build custom plugins
```
cd custom_plugins
mkdir build && cd build
cmake ..
make
```
## Run deepstream-triton
```
LD_PRELOAD=<path-to-plugins-.so> python3 run_scrfd_triton.py <path-to-sample-mp4>
```
