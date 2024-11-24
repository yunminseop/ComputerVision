# Perception
Perception code(Sliding Window) provided by Lee Seung Min

![Screenshot from 2024-11-24 22-41-22](https://github.com/user-attachments/assets/5ecaad31-d9cd-456a-900f-ef62efcd4803)

Main scripts: Sliding Window/ Sobel Filter/ BirdEyeView  
"perception.mp4" is in data folder.

If you got an error message like ***"H.264 (Constrained Baseline profile) decoder is required to play the file, but is not installed."***,  
you should execute following commands on your terminal.

$ sudo apt update  
$ sudo apt install ubuntu-restricted-extras  
$ sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  

---
perception.py includes cv2 library. I recommend you execute this file in virtual environment including "opencv-python" and **keep it without any dependency crash.**  
For example, you may have a hard time if you install "labelme" in the virtual environment. It causes crash.
