# RVM_onnx_compose
RobustVideoMatting and background composing in one model by using onnxruntime.
# Usage
```
pip install -r requirements.txt
python infer_cam_onnx_freeze.py
```
Press key `1~4` for changing 4 background images in folder `backgrounds`, press key `5` for using video as background, press `0` for green background, press `q` for raw web-cam stream. 
# To do
Solve memory leak, which may caused by `cap.read()`