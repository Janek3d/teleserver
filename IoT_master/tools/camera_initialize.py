import cv2
from configparser import ConfigParser


class Camera:
    def __init__(self, file):
        config = ConfigParser()
        config.read(file)
        self.login = config.get('camera_address','login')
        self.password = config.get('camera_address','password')
        self.ip_address = config.get('camera_address','ip_address')
        self.channel = config.get('camera_address', 'channel')
        self.address = (f'rtsp://{self.login}:{self.password}@{self.ip_address}'
                        f'/cam/realmonitor?channel={self.channel}&subtype=0')
        self.cam = cv2.VideoCapture(self.address)

    def change_channel(self, new_channel):
        if new_channel == 'thermal' or new_channel == '2' or new_channel == 2:
            self.channel = '2'
        else:
            self.channel = '1'
        self.address = (f'rtsp://{self.login}:{self.password}@{self.ip_address}'
                        f'/cam/realmonitor?channel={self.channel}&subtype=0')
        self.cam = cv2.VideoCapture(self.address)

    def get_picture(self):
        self.ret, self.img = self.cam.read()
        return self.ret, self.img


if __name__ == '__main__':

    kamera_class = Camera('config.ini')
    cv2.namedWindow('Camera')
    print(kamera_class.address)
    channels = ['1', '2']
    i = 0

    while True:
        ret, img = kamera_class.get_picture()
        if not ret:
            break
        cv2.imshow('Camera', img)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            kamera_class.change_channel(channels[i % 2])
            i = i+1
    kamera_class.cam.release()
    cv2.destroyAllWindows
