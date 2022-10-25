# deepstream-face-recognition
Face detection -> alignment -> feature extraction with deepstream
## 1. Requirements
- Deepstream 6.1 (recommend using docker)

## 2. Installation
### 2.1 Build docker
```
docker build -t face_recog_ds6.1 - < Dockerfile
```
### 2.2 Run docker (remember to mount)
```
docker run --rm -it -v<path-to-source>:<path-to-destination> --gpus all face_recog_ds6.1:latest
```
### 2.3 Compile plugins (run inside docker)
- Compile custom **NMS** plugin (**batchedNMSDynamic_TRT** with landmark)
```
export CUDA_VER=11.4
cd plugins/nms
mkdir build && cd build
cmake ..
make
```
- Compile custom **gst-nvinfer** plugin
```
cd gst-nvinfer
make
make install
```
## 3. Run demo
- object detection -> face detection (from cropped person image) -> face embedding
```
LD_PRELOAD=<path-to-NMS-plugin> python main.py file:<path-to-video-input>
```
- face detection (full frame) -> face embedding
```
LD_PRELOAD=<path-to-NMS-plugin> python main_ff.py file:<path-to-video-input>
```

## 4. To do
- [x] Add cropped/fullframe pipeline for face
- [ ] Fix 1-batchsize error (we have some bugs with custom NMS plugin, only have true output with batchsize=1)
- [ ] Improve pipelines robustness 
- [ ] Add cropped pipeline for license plate
