import cv2

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture('../../Images/street.mp4')
ok, frame = video.read()

bbox = cv2.selectROI(frame)
print(bbox)

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()

    if not ok:
        break

    ok, bbox = tracker.update(frame)
    print(bbox)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, 'Tracker failed', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break