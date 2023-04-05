import cv2
faceCascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
while True:
    #Taking the live input
    ret, img = cap.read()
    #Converting BGR image to Grayscale Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detecting the face from the Image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
        )
    #Showing boxes around the face and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)
            #Showing the Output Image
        cv2.imshow('Press Escape to exit.', img)
    #Creating a close button for the app by pressing Escape key
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
#After the application is closed, releasing the access of the camera
cap.release()
cv2.destroyAllWindows()