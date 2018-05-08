import cv2
import numpy as np
import glob as gb
import os

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

def get_sblk(img_mat, sub_size):
    img_size = img_mat.shape

    sblk = list()
    ln = list()

    row_temp = list()
    row_ln_temp = list()

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
    
    return sblk, ln

def get_target_max(ilcm):
    target_max = [0,0]
    temp_ilcm = 0
    for i in range(ilcm.shape[0]):
        for j in range(ilcm.shape[1]):
            if ilcm[i,j] > temp_ilcm:
                target_max = [i,j]
                temp_ilcm = ilcm[i,j]
                #print("target position:%d\t%d" % (i, j))
    return np.mat(target_max)

def get_th(ilcm):
    mu = np.mean(ilcm)
    sigma = np.std(ilcm)
    k = 20
    return mu + sigma*k

def get_target_pos(target_list):
    pos = target_list * step_size
    return pos


def draw_rectangle(pos, img, output):
    image = img.copy()
    rec_size = 10
    new_pos = pos.tolist()
    #print(new_pos)
    if new_pos[0]:
        for i in range(len(new_pos)):
            cv2.rectangle(image, (new_pos[i][0]-rec_size+20,new_pos[i][1]+rec_size+20), (new_pos[i][0]+rec_size+20,new_pos[i][1]-rec_size+20), (0,255,0), 1)
            #cv2.imshow('input image',img)
            #cv2.imshow('output image',image)

    else:
        print("Detection Failed")
    print('output file: '+output)
    cv2.imwrite(output, image)
    return

def img_process(input_path):
    file_name = os.path.basename(input_path)
    print('loading %s' % input_path)
    img = cv2.imread(input_path)


    diff1 = cv2.GaussianBlur(img, (5,5), 1)
    diff2 = cv2.GaussianBlur(img, (3,3), 1)
    dog = np.mat(cv2.cvtColor(diff1 - diff2,cv2.COLOR_BGR2GRAY))
    
    dog_shrink = dog[10:246, 10:246]

    #cv2.imshow("DoG", dog)
    #cv2.imshow("DoG_shrink", dog_shrink)
    #cv2.imshow("Origin", img)


    sblk , ln = get_sblk(dog_shrink, sub_size)
    ilcm = get_ilcm(ln, sblk)
    th = get_th(ilcm)
    target_list = get_target_max(ilcm)
    pos = get_target_pos(target_list)
    output = output_path+file_name
    draw_rectangle(pos, img, output)
    
    return



global sub_size
sub_size = 16
global step_size
step_size = int(sub_size/2)
global output_path
output_path = './output_DoG/' 

