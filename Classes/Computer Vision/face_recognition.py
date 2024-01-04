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
        image_np = np.array(image, 'uint8')
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

        faces.append(image_np)
        ids.append(id)

    return np.array(ids), faces

ids, faces = image_data()

print(ids, '===================================================================================================================\n')
print(faces, '===================================================================================================================\n')

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, ids)
lbph.write('LBPH_classifier.yml')