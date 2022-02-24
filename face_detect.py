import cv2
import imutils


def faceDetector(file, upload_folder, dest_folder):
    face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_img = cv2.imread(upload_folder + file)
    face_img = imutils.resize(face_img, width=580)

    # grayscale img
    face_grey_scale = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # detect face
    face_detected = face_cas.detectMultiScale(face_grey_scale, 1.1, 4)

    # process img
    count = 0
    for (x, y, width, height) in face_detected:
        cv2.rectangle(face_img, (x, y), (x+width, y+height), (255, 0, 255), 3)
        count += 1

    # write result
    cv2.imwrite(dest_folder + file, face_img)

    return count

# faceDetector('face.jpg', 'static/uploads/', 'static/faces/')
