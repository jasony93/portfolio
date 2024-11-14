# Portfolio

face_verification.c
- Used GhostFaceNet for face verification application. 
- Takes around 500ms/frame on 1.3TOPS RTOS environment.
- The file shows preprocess -> model inference -> postprocess -> vector similarity calculation


yolov6_dms.c
- Used Yolo version6 for object detection of driver monitoring system.
- Takes around 100ms/frame on 1.3TOPS RTOS environment.
- The file shows preprocess -> model inference -> postprocess