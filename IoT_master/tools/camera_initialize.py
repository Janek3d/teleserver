import cv2

class Camera:
    def __init__(self):
        self.login = 'admin'
        self.password = 'qwerty123456'
        self.ip_address = '192.168.1.108'
        self.channel = '1'
        self.adres_part_1 = 'rtsp://'+self.login+':'+self.password+'@'+self.ip_address+'/cam/realmonitor?channel='
        self.adres_part_2 = '&subtype=0'
        
    def camera(self,channel='thermal'):
        if channel == 'thermal':
            self.adres = self.adres_part_1+'2'+self.adres_part_2
        else:
            self.adres = self.adres_part_1+'1'+self.adres_part_2
        #self.cam = cv2.VideoCapture(self.adres)
        self.cam = cv2.VideoCapture(self.adres)
        self.ret, self.img = self.cam.read()
        return self.ret, self.img


if __name__=='__main__':

    kamera_class = Camera()
    cv2.namedWindow('Camera')
    #cv2.namedWindow('Camera')
    while True:
        ret, img = kamera_class.camera()
        if not ret:
            break
        cv2.imshow('Camera',img)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    kamera_class.cam.release()
    cv2.destroyAllWindows
