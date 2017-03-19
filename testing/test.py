import numpy as np
import cv2

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

src = np.zeros((30*80, 30*145, 3), dtype=np.uint8)
sub_h = 80
sub_w = 145

colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]

class_n = 7

predictions= (np.random.rand(30,30)*(class_n)).astype(int)

for prediction_row_i, prediction_row in enumerate(predictions):
    for prediction_col_i, prediction in enumerate(prediction_row):
        color = colors[prediction]
        cv2.rectangle(src, (prediction_col_i*sub_w, prediction_row_i*sub_h), (prediction_col_i*sub_w+sub_w, prediction_row_i*sub_h+sub_h), color, -1)

disp_img_fullscreen(src)
