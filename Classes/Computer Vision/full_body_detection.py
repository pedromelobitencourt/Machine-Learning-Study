import cv2

image = cv2.imread('../../Images/people.jpg')

# show the image
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

full_body_cascade_path = '../../Cascades OpenCV/fullbody.xml'
full_body_detector = cv2.CascadeClassifier(full_body_cascade_path)

# Check if the classifier is loaded successfully
if full_body_detector.empty():
    print("Error: Could not load face cascade classifier.")
else:
    # Turn the image into a gray image, because it is faster to process it
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the gray image
    # cv2.imshow('Gray image', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    detections = full_body_detector.detectMultiScale(gray_image, scaleFactor=1.1, minSize=(50, 50))
    print(detections)

    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # it couldn't detect one person
    cv2.imshow('Body detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()