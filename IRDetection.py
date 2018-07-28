import cv2
import numpy as np
import glob as gb
import os
import Filter as fi
  


#img_path = gb.glob(".\\data_set\\*.bmp")
img_path = gb.glob(".\\data_set2\\*.jpg")
#fi.img_process('./data_set/2dim88.bmp')
#fi.img_process('./data_set2/0692.jpg')
for path in img_path:
    #bs.img_process(path)
    #max.img_process(path)
    #di.img_process(path)
    #dog.img_process(path)
    fi.img_process(path)

    pass
    

cv2.waitKey()
cv2.destroyAllWindows()

#if __name__ == '__main__':
#    main()