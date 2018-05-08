import cv2
import numpy as np
import glob as gb
import os
import IRDetection_baseline as bs
import IRDetection_max as max
import dialateDetection as di
import DoGIRDetection as dog
  


img_path = gb.glob(".\\data_set\\*.bmp")
dog.img_process('./data_set/2dim88.bmp')
for path in img_path:
    #bs.img_process(path)
    #max.img_process(path)
    #di.img_process(path)
    dog.img_process(path)

    pass
    

cv2.waitKey()
cv2.destroyAllWindows()

#if __name__ == '__main__':
#    main()