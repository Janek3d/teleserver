#!/bin/sh

# Installation of YOLOv3
git clone https://github.com/Janek3d/yolov3.git
if [ $? -ne 0 ]; then
	echo "Could not install."
	exit 1
fi

echo "YOLOv3 has been installed successfully."
exit 0
