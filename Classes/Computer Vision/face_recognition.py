import numpy as np
import os
import cv2
from PIL import Image

def image_data():
    paths = [os.path.join('../../Databases/yalefaces/train', f) for f in os.listdir('../../Databases/yalefaces/train')]
    faces = []
    ids = []

    for path in paths:
        image = Image.open(path).convert('L') # L: one image channel
        np_image = np.array(image, 'uint8')
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

        faces.append(np_image)
        ids.append(id)

    return np.array(ids), faces

ids, faces = image_data()

print(ids, '===================================================================================================================\n')
print(faces, '===================================================================================================================\n')

## Creating a classifier
# lbph = cv2.face.LBPHFaceRecognizer_create()
# lbph.train(faces, ids)
# lbph.write('LBPH_classifier.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('LBPH_classifier.yml')

test_image = '../../Databases/yalefaces/test/subject10.sad.gif'
image = Image.open(test_image).convert('L')
np_image = np.array(image, 'uint8')

print(np_image, '\n\n\n')

forecast_id, _ = recognizer.predict(np_image)
print('forecast id:', forecast_id)

right_id = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))
print('right id:', right_id, '\n')

cv2.putText(np_image, 'P:' + str(forecast_id), (30, 100 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(np_image, 'R:' + str(forecast_id), (30, 100 + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.imshow('Prediction', np_image)
cv2.waitKey(0)
cv2.destroyAllWindows()