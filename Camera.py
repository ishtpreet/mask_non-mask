import cv2

# Class for VideoCamera
class VideoCamera(object):
    # Constructor for capturing video/live feed
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    # Destructor for releasing the video/live feed
    def __del__(self):
        self.video.release()

    # Reading the frame from the live feed
    def get_frame(self):
        _, fr = self.video.read()
        img = cv2.flip(fr, 1)
        return img


