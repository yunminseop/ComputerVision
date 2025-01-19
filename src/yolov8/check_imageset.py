# WARNING #
# use opencv venv instead of yolo #

import cv2
import os

filepath = '****'
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Video is unavailable:", filepath)
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length:", length)
print("width:", width)
print("height:", height)
print("fps:",fps)


try:
    if not os.path.exists(filepath[:-4]):
        os.makedirs(filepath[:-4])

except OSError:
    print("Error: Creating directory. " + filepath[:-4])

    

