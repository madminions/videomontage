import time
import Constants
from Animations import Animations
import cv2

start = time.time()
size = Constants.SIZE
frame_rate = Constants.FRAME_RATE
anim = Animations()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
file_name = "output.avi"
images = ['images/a.jpg','images/b.jpg','images/c.jpg','images/d.jpg']

out = cv2.VideoWriter(file_name, fourcc, frame_rate, size)

for iimgs, img_path in enumerate(images):
    img = cv2.imread(img_path)
    print(img_path)
    (h,w) = img.shape[:2]

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=Constants.WHITE)
    blur = anim.get_blur_img(img,size)

    for i in anim.img_animation_zoom_in(img, blur):
        out.write(i)

out.release()
