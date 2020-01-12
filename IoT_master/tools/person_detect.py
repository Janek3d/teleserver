import argparse
from sys import platform

from tools.yolov3.models import *  # set ONNX_EXPORT in models.py
from tools.yolov3.utils.datasets import *
from tools.yolov3.utils.utils import *

class detection:
    """
    This class is used to detect people in the image and find  
    coordinates of bounding boxes of these people, mainly x center and y center.

    """
    def __init__(self):
        """
        Initialization of the model used to detect people.
        """
        img_size = 416
        #img_size = (320, 192) if ONNX_EXPORT else img_size2  # (320, 192) or (416, 256) or (608, 352) for (height, width)

        weights = 'tools/yolov3/weights/yolov3.weights'
        source = 'rtsp://admin:qwerty123456@192.168.1.108/cam/realmonitor?channel=2&subtype=0'
        cfg = 'tools/yolov3/cfg/yolov3.cfg'
        self.out = 'yolov3/output'
        self.halff = True

        self.webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        devicee = ''
        self.device = torch_utils.select_device(device=devicee)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder

        # Initialize model
        self.model = Darknet(cfg, img_size)
        _ = load_darknet_weights(self.model, weights)
        # Eval mode
        self.model.to(self.device).eval()

        # Half precision
        half = self.halff and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        # Set Dataloader
        if self.webcam:
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=img_size, half=half)

        # Get names and colors
        namess = 'tools/yolov3/data/coco.names'
        self.names = load_classes(namess)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, view_img=False):
        """
        Detect people and based on their coordinates in the image, determines if table is free or occupied.
        Returns list of 1 - occupied or 0 - free.

        :param view_img: Display image from camera with bounding boxes.
        :type file: boolean

        :return occupancy: List of int values
        :rtype occupancy: int
        """
        
        # Run inference
        t0 = time.time()
        for path, img, im0s, vid_cap in self.dataset:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img)[0]

            if self.halff:
                pred = pred.float()

            # Apply NMS
            conf_thres = 0.3
            nms_thres = 0.5
            pred = non_max_suppression(pred, conf_thres, nms_thres)

            # Process detections
            occupancy=[0,0,0,0,0,0]

            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s
                if view_img:
                    cv2.rectangle(im0,(0,0),(427,360),(0,255,0),thickness=3)
                    cv2.rectangle(im0,(428,0),(853,360),(0,255,0),thickness=3)
                    cv2.rectangle(im0,(854,0),(1280,360),(0,255,0),thickness=3)
                    cv2.rectangle(im0,(0,361),(427,720),(0,255,0),thickness=3)
                    cv2.rectangle(im0,(428,361),(853,720),(0,255,0),thickness=3)
                    cv2.rectangle(im0,(854,361),(1280,720),(0,255,0),thickness=3)

                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Print time (inference + NMS)
                    #print('%sDone. (%.3fs)' % (s, time.time() - t))

                    # Write results
                    
                    for *xyxy, conf, cls in det:
                        xavg = (int(xyxy[0])+int(xyxy[2]))/2
                        yavg = (int(xyxy[1])+int(xyxy[3]))/2
                        if int(cls)==0:
                            #print("Pozycja obietku x:", xavg, 'y: ', yavg)
                            if(xavg<427 and yavg<361):
                                if view_img:
                                    cv2.rectangle(im0,(0,0),(426,359),(0,0,255),thickness=3)
                                occupancy[0]=1

                            if(xavg>427 and yavg<361 and xavg<853):
                                if view_img:
                                    cv2.rectangle(im0,(428,0),(852,359),(0,0,255),thickness=3)
                                occupancy[1]=1
        
                            if(xavg>853 and yavg<361):
                                if view_img:                                
                                    cv2.rectangle(im0,(854,0),(1280,359),(0,0,255),thickness=3)
                                occupancy[2]=1

                            if(xavg<427 and yavg>=361):
                                if view_img:
                                    cv2.rectangle(im0,(0,361),(426,719),(0,0,255),thickness=3)
                                occupancy[3]=1

                            if(xavg>427 and yavg>=361 and xavg<853):
                                if view_img:
                                    cv2.rectangle(im0,(428,361),(852,719),(0,0,255),thickness=3)
                                occupancy[4]=1

                            if(xavg>853 and yavg>=361):
                                if view_img:
                                    cv2.rectangle(im0,(854,361),(1279,719),(0,0,255),thickness=3)
                                occupancy[5]=1

                        if view_img:  # Add bbox to image
                            if int(cls)==0:
                                label = '%s %.2f' % (self.names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

                #print(bboxy)

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
            break
        return(occupancy)


if __name__ == '__main__':
    
    det = detection()

    with torch.no_grad():
        det.detect()