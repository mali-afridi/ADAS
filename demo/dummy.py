import numpy as np
import cv2
import time
# import pyautogui
# from directkeys import PressKey,ReleaseKey, W, A, S, D
# from draw_lanes import draw_lanes
# from grabscreen import grab_screen
def roi(img, vertices):
    
    #blank mask:
    mask = np.zeros_like(img)   
    # import pdb; pdb.set_trace()
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, (255,255,255) )
    # import pdb; pdb.set_trace()
    indices = np.where(mask != (255, 255, 255))
    # import pdb; pdb.set_trace()
    img[indices] = 0
    # cv2.bitwise_and(img, mask)
    cv2.imwrite('roi2.jpg',img)

    
    #returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked
vertices = np.array([[0,0],[1640,0],[1640,450],[0,450]
                        ], np.int32)


processed_img = cv2.imread('/home/sami/Desktop/Code/CLRerNet/demo/00020.jpg')
# import pdb; pdb.set_trace()
processed_img = roi(processed_img, [vertices])
cv2.imwrite('roi.jpg',processed_img)