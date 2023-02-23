import cv2
import numpy as np
import time


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.colorList = None
        self.classesList = None
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Initialize network
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

        print(self.classesList)

    # to open video
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        # if no video shoe error
        if (cap.isOpened()==False):
            print("Error opening file...")
            return
        # reading video if not error
        (success, image) = cap.read()

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.5)

            # converting confi & bbox to list
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            # eliminate overlapping of bounding boxes
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}: {:.4f}".format(classLabel, classConfidence)

                    # unpack bounding box to get its x & y co-ordinates
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=2)
                    cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    # beautify bounding boxes
                    lineWidth = min(int(w * 0.3), int(h * 0.3))
                    # top-left corner
                    cv2.line(image, (x, y), (x+lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y+lineWidth), classColor, thickness=5)
                    # top-right corner
                    cv2.line(image, (x+w, y), (x+w-lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x+w, y), (x+w, y+lineWidth), classColor, thickness=5)

                    # bottom-left corner
                    cv2.line(image, (x, y+h), (x + lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x, y+h), (x, y+h - lineWidth), classColor, thickness=5)
                    # bottom-right corner
                    cv2.line(image, (x+w, y+h), (x+w - lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x+w, y+h), (x+w, y+h - lineWidth), classColor, thickness=5)

            # showing frame
            cv2.imshow("Output", image)

            # to break out of the loop
            key = cv2.waitKey(1) & 0xFF
            if key== ord('q'):
                break

            # to capture next frame
            (success, image) = cap.read()

        # once loop break, we destroy all cv2 windows
        cv2.destroyAllWindows()


