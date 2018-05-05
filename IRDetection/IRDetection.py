import cv2
im = cv2.imread("./data_set/1.bmp")  #读取图片
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


cv2.imshow("image", im_gray)
cv2.waitKey()
#cv2.show()