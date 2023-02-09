import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")
def sarphen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return sharp
def lin_blur(image):
    sharp = cv2.medianBlur(image,3)
    return sharp


def blur_finder():
    val1 = 0
    pos = 0
    for x in range(5):
        img = cv2.imread('C:/Users/masdaq/Desktop/ocv/ultralytics-main/opencv_frame_' + str(x) + '.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val0 = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(val0)
        if val0 >= val1:
            val1 = val0
            pos = x
            if (an0 == "y" or an1 == "y"):

                if (an0 == "y" and an1 == "y"):
                    img = sarphen(img)
                    img = lin_blur(img)
                elif(an0 == "y" and an1 == "n"):
                    img = sarphen(img)
                else:
                    img = lin_blur(img)
                img = cv2.imwrite('C:/Users/masdaq/Desktop/ocv/ultralytics-main/opencv_frame_' + str(pos) + '.png', img)



    predictions = model.predict(source='opencv_frame_' + str(pos) + '.png', show=wns, conf=0.5, device = dvc )
    #model.predict(source=0, show=wns, conf=0.5, device=dvc)#too see live video
    inf0 = ""
    inf1 = ""
    inf2 = ""
    temp0 = 0
    temp1 = 0
    for x in range(len(predictions)):
        print("[T]")
        # print(predictions[x])
        if x == 0:
            # temp1 = predictions[x].find("[")
            inf0 = predictions[x]
            inf0 = str(inf0).split()
            temp1 = inf0.index("shape:")
            inf0 = str(inf0[temp1 + 1:temp1 + 3])
            inf0 = [int(i) for i in inf0 if i.isdigit()]
            print("*************************************")
            print("!!Number of people:    "+str(inf0[0]))
            print("*************************************")
        elif x == 1:
            inf1 = predictions[x]
    print(str(pos) +" is the most clear one with "+str(val1))

an0 =  ""
while (an0 == "y" or an0 == "n") == False :
    an0 = input("Would you like to apply sharpening? we dont usally recommond this on normal weathers it is recommended for rainy and foggy weathers y/n ")
an1 =  ""
while (an1 == "y" or an1 == "n") == False :
    an1 = input("Would you like to apply linearfilter? against noise the cheaper your camera more you would like to aplly this y/n ")
an2 =  ""
while (an2 == "y" or an2 == "n") == False :
    an2 = input("Does your pc supporrt CUDA and is it set up? y/n ")
an3 =  ""
while (an3 == "y" or an3 == "n") == False :
    an3 = input("would you like to see images with detection? y/n ")
if an2 == "y":
    dvc = "gpu"
else:
    dvc ="cpu"

if an3 == "y":
    wns = "True"
else:
    wns ="False"


print("please wait for camera to start ")
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:#esc
        print("Escape hit, closing...")
        break
    else:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img = cv2.imread('C:/Users/masdaq/Desktop/ocv/ultralytics-main/opencv_frame_'+str(img_counter)+'.png')
        #cv2.imshow('sample image', img)
        img_counter += 1
        #time.sleep(1)
    if img_counter == 5:
        img_counter = 0
        blur_finder()

cam.release()

cv2.destroyAllWindows()

#blur_finder()


