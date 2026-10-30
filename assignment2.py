import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")
input_video = "dance.mp4"
output_video = "skeleton_output.mp4"
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
     print("Error: Cannot open input video.")
     exit()
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps <= 0:
    fps = 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)
skeleton_connections = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]
frame_number = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    black_frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )
    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )
    for result in results:

        if result.keypoints is None:
            continue

        if result.keypoints.xy is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            confidences = result.keypoints.conf.cpu().numpy()
        else:
            confidences = np.ones(
                (len(keypoints), 17)
            )

        for person_index, person in enumerate(keypoints):

            scores = confidences[person_index]

            for start, end in skeleton_connections:

                if scores[start] > 0.25 and scores[end] > 0.25:

                    x1, y1 = map(int, person[start])
                    x2, y2 = map(int, person[end])

                    cv2.line(
                        black_frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        thickness=3,
                        lineType=cv2.LINE_AA
                    )

            for i, point in enumerate(person):

                if scores[i] > 0.25:

                    x, y = map(int, point)

                    cv2.circle(
                        black_frame,
                        (x, y),
                        radius=4,
                        color=(255, 255, 255),
                        thickness=-1,
                        lineType=cv2.LINE_AA
                    )
    out.write(black_frame)
    cv2.imshow(
        "YOLO Human Pose Estimation",
        black_frame
    )

    print("Processing frame:", frame_number)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

print("Pose estimation completed!")
print("Output saved as:", output_video)
