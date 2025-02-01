import cv2

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'width = {width}, height = {height}')


fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while True:
    ret, frame = capture.read()

    key = cv2.waitKey(30)
    if key == ord('r') and record == False:
        record = True
        video = cv2.VideoWriter("test_driving.avi",fourcc, 20.0, (width, height), isColor=True)

    elif key == ord('r') and record == True:
        record = False
        video.release()
    
    elif key == ord('q'):
        break
    
    if record == True:
        video.write(frame)
        cv2.circle(img=frame, center = (620, 15), radius = 4, color = (0, 0, 255), thickness = 3)
        

    cv2.imshow("test_driving", frame)

capture.release()
cv2.destroyAllWindows()
