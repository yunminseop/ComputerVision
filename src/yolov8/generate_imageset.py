# WARNING #
# use opencv venv instead of yolo #

import cv2

filepath = "****"
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Video is unavailable:", filepath)
    exit(0)


length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

# you have to change this number. (it's related to file names.)
count = 203

while(video.isOpened()):
    ret, image = video.read()

    # saving images per 1 sec #
    if (int(video.get(1)) % int(fps)) == 0:
        cv2.imwrite("/home/ms/my_yolo_data/dataset/images/frame%d.jpg" % count, image)
        # print("/home/ms/my_yolo_data/dataset/images/frame"+str(count)+".jpg")
        
        count += 1

    # saving images per 1 frame #  
    # cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % video.get(1), image)
    # print("Saved frame number:", str(int(video.get(1))))

    if int(video.get(1)) == length:
        video.release()
        break
