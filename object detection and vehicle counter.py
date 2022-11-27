import cv2
import numpy as np

#Web Camera
cap = cv2.VideoCapture('assets/video.mp4')
#declaration
min_width_rectangle = 80 #min width rectangle
min_height_rectangle = 80 #min height rectangle

count_line_position = 600  #position of line
# Initialize Subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) #background subtraction-->eliminating the background from video

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
offset = 6 #Allowable error b/w pixel
counter = 0

while True:
    ret, video = cap.read()  #reads video
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY) # converts video color
    blur = cv2.GaussianBlur(gray, (3,3), 5) #video smoothing preventing from noise

# Applying on each frame
    vid_sub = algo.apply(blur) #algo applies with gblur
    dilat = cv2.dilate(vid_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countershape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(video, (25,count_line_position),(1600,count_line_position),(200,0,0), 3)

# creating rectangle around vehicle
    for (i, c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rectangle) and (h>= min_height_rectangle)
        if not val_counter:
            continue
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(video,"Vehicle No:   " + str(counter), (x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(video, center, 4, (0,0,255), -1)
# for counting the vehicles
        for (x,y) in detect:
            if y<(count_line_position + offset) and  y>(count_line_position - offset):
                counter+=1
                cv2.line(video, (25,count_line_position),(1200,count_line_position),(0,127,255), 3)
                detect.remove((x,y))

                print("Vehicle No: "+ str(counter))

    cv2.putText(video,"Vehicle Count : " + str(counter), (5,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,25),5)

    cv2.imshow('Vehicle _Counter',video) # shows video

    if cv2.waitKey(40) == ord("g"):
        break
cap.release()
cv2.destroyAllWindows() # closes window
