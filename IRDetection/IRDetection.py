import cv2
import numpy as np

def get_ln(matrix):
    ln = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]>ln:
                ln = matrix[i,j]
    return ln

def get_ilcm(ln, sblk):
    ilcm_row = list()
    for row in range(1,sblk.shape[1]-1):
        ilcm_col = list()
        for col in range(1,sblk.shape[0]-1):
            ilcm_temp = list()
            for i in range(3):
                for j in range(3):
                    if (i!=1 and j!=1):
                        ilcm_temp.append(ln[row, col]*sblk[row, col]/sblk[row+1-i,col+1-j])
            ilcm_col.append(min(ilcm_temp))
        ilcm_row.append(ilcm_col)

    ilcm_mat = np.mat(ilcm_row)
    return ilcm_mat

fn = "data_set/1.bmp"

print('loading %s' % fn)
img = cv2.imread(fn)
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sub_size = 64
step_size = int(sub_size / 2)
img_size = img_grey.shape
img_mat = np.mat(img_grey)

sblk = list()
ln = list()
row_temp = list()
row_ln_temp = list()
#print('width:%d\nheight:%d' % (img_size[1],img_size[0]))
#print((img_size[0]-sub_size)/step_size)

for row in range(img_size[0]):
    col_temp = list()
    col_ln_temp = list()
    for col in range(int((img_size[1]-sub_size)/step_size)):
        temp_grey = 0
        max_grey = 0
        for i in range(sub_size):
            temp_grey += img_mat[row,col*step_size+i]
            if img_mat[row,col*step_size+i] > max_grey:
                max_grey = img_mat[row,col*step_size+i]
        col_temp.append(temp_grey/sub_size)
        col_ln_temp.append(max_grey)
    row_temp.append(col_temp)
    row_ln_temp.append(col_ln_temp)

temp_mat = np.transpose(np.mat(row_temp))
temp_ln = np.transpose(np.mat(row_ln_temp))

for row in range(int((img_size[0]-sub_size)/step_size)):
    col_temp = list()
    col_ln_temp = list()
    for col in range(int((img_size[0]-sub_size)/step_size)):
        temp_grey = 0
        max_grey = 0
        for i in range(sub_size):
            temp_grey += temp_mat[row,col*step_size+i]
            if temp_mat[row,col*step_size+i] > max_grey:
                max_grey = temp_mat[row,col*step_size+i]
        col_temp.append(temp_grey/sub_size)
        col_ln_temp.append(max_grey)
    sblk.append(col_temp)
    ln.append(col_ln_temp)

sblk = np.transpose(np.mat(sblk))
ln = np.transpose(np.mat(sblk))

print(sblk)
#print(sblk.shape)
print(ln)
#print(ln.shape)

ilcm = get_ilcm(ln, sblk)
print(ilcm)

for i in range(ilcm.shape[0]):
    for j in range(ilcm.shape[1]):
        if ilcm[i,j] > sblk[i,j]:
            print("target position:%d\t%d" % (i, j))



sp = img.shape
print(sp)



# 获取图像大小
sz1 = sp[0]
sz2 = sp[1]
print('width:%d\nheight:%d' % (sz2,sz1))
# 创建一个窗口显示图像
#cv2.namedWindow('img')
#cv2.imshow('img_orig',img)
# 复制图像矩阵，生成与源图像一样的图像，并显示

#cv2.imshow('img_grey',img_grey)
cv2.waitKey()
cv2.destroyAllWindows()

#if __name__ == '__main__':
#    main()