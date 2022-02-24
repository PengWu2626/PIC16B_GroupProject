import cv2
import imutils

UPLOAD_FOLDER = 'static/uploads/'
DEST_FOLDER = 'static/faces/'

face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_img = cv2.imread(UPLOAD_FOLDER + 'face.jpg')
face_img = imutils.resize(face_img, width=580)

# grayscale img
face_grey_scale = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
# detect face
face_detected = face_cas.detectMultiScale(face_grey_scale, 1.1, 4)

# process img
for (x, y, w, h) in face_detected:
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# write result
cv2.imwrite(DEST_FOLDER + "result.png", face_img)
