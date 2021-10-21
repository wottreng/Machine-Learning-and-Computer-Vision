import cv2
import numpy
from time import time, sleep
import os

#stream video input
stream = False
bias = 0.55
textColor = (255, 250, 250) #(b,g,r)  0-255
boxColor = (0, 200, 0)

pictureNames = os.listdir(f"{os.getcwd()}/dataset")

print("Once started press 'q' to quit.")
# laptop camera
vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture(vs)
width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"video h,w: {width}, {height}")
if not (vs.isOpened()):
    print("Could not open video device")

vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

trainingPics=[]
for pic in pictureNames:
    pictureArray = cv2.imread(f'{os.getcwd()}/dataset/{pic}')
    w = pictureArray.shape[1]
    h = pictureArray.shape[0]
    trainingPics.append([pictureArray,w,h,pic])



fps_time = time()
while True:
    # Grab the frame from the threaded video stream.
    ret, frame = vs.read()
    if ret:
        scr = numpy.array(frame)
        scr_remove = scr[:, :, :3] # Cut off alpha

        for datapic in trainingPics:
            match = cv2.matchTemplate(scr_remove, datapic[0], cv2.TM_CCOEFF_NORMED)
            # match = cv2.matchTemplate(scr, datapic[0], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match)

            #print(f"Max Val: {max_val} Max Loc: {max_loc}")
            if max_val > bias:
                pt0 = max_loc  # tuple
                pt1 = max_loc[0] + datapic[1], max_loc[1] + datapic[2]
                cv2.rectangle(frame, pt0, pt1, boxColor, 2)  # color:(b,g,r)
                x = max_loc[0]  #
                y = max_loc[1]  #
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, f"{round(max_val, 2)}:{datapic[3]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, textColor,2)


        cv2.imshow('web cam', frame)

    sleep(.10)
    if not stream:
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    #print('FPS: {}'.format(1 / (time() - fps_time)))
    fps_time = time()

vs.release()
cv2.destroyAllWindows()
