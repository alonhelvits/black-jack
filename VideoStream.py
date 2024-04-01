from threading import Thread
import cv2

class VideoStream:
    """Camera object"""

    def __init__(self, resolution=(640, 480), framerate=30, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        self.frame = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
            else:
                self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
