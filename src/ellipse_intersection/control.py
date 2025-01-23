from pursuit import *
import cv2
import numpy as np
from ultralytics import YOLO
import torch 
import torch.nn as nn
import logging



class Control(VideoProcessor):
    def __init__(self, video_path, model):
        super().__init__(video_path=video_path, model=model)
    
    def steering(self, slope):
        print(slope)






if __name__ == "__main__":
    video_path = '/home/ms/Downloads/video_output5.mp4'
    checkpoint_path = '/home/ms/Downloads/best.pt'

    model = YOLO(checkpoint_path, verbose=False)

    controller = Control(video_path, model)

    controller.process_video()