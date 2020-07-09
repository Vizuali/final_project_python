# Vizuali Final Project - Face recognition in python



[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This project uses face recognition from this [repository](https://github.com/ageitgey/face_recognition).

# Requirements
  - Ubuntu with NVIDIA GPU
  - Python 3.3+
  - cmake

# Installation

  - Install dlib with GPU support
```sh
 git clone https://github.com/davisking/dlib.git
 cd dlib
 mkdir build
 cd build
 cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
 cmake --build .
 cd ..
 python3 setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
```
  - Install face-lib
```sh
 python3 pip install dlib
```
  - If you want to use the project without GPU support:

```sh
 python3 pip install -r requirements.txt
```
# Usage

Just execute the camera_recognition file:
```sh
 python3 camera_recognition.py
```


# Results

### Face recognition and blur:
The coordinates of the face recognized are in the down left corner. The program recongnizes the face and make a blur on it.

![Face recognition](https://i.imgur.com/3PIKelI.png)

Videos

https://s7.gifyu.com/images/2020-07-08-16-54-02.gif

https://s7.gifyu.com/images/2020-07-08-16-56-59.gif

https://s7.gifyu.com/images/2020-07-08-16-47-15.gif

### Detected faces.
![Detected faces](https://i.imgur.com/9Hs7Iu0.jpg)

### List of coordinates of faces detected (No GPU)
Video length: 4s - 133 frames 240p
Processing time: 5 minutes
Ubuntu 18.04 2 Cores 4 Gb RAM (VM)

![Detected faces in console](https://i.imgur.com/2B0XKsQ.png)

Not recommended to run the program only on CPU.



License
----

MIT
