import cv2

image = cv2.imread('../../Images/workplace.jpg')

## show the image
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Use forward slashes in the file path
face_cascade_path = '../../Cascades OpenCV/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(face_cascade_path)

# Check if the classifier is loaded successfully
if face_detector.empty():
    print("Error: Could not load face cascade classifier.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## show the gray image
    # cv2.imshow('Image', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('oioioioioi')
    detections = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minSize=(30, 30)) # scaleFactor: zooming in, minSize: face minimum size

    # array of arrays
    # detections[0]: [ face initial position x (pixel), face initial position y(pixel), x face length, y face length ]
    print(detections) 

    for(x, y, l, a) in detections:
        cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2) # image, (x, y), (x + l, y + a) color, border size
    
    cv2.imshow('Face detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()