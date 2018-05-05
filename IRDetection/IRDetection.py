import cv2
import numpy as np
import glob as gb

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

def get_target_th(ilcm, th):
    target_list = list()
    for i in range(ilcm.shape[0]):
        for j in range(ilcm.shape[1]):
            if ilcm[i,j] > th:
                target_list.append([i,j])
                #print("target position:%d\t%d" % (i, j))
    return np.mat(target_list)

def get_target(ilcm):
    target_list = list()
    for i in range(ilcm.shape[0]):
        for j in range(ilcm.shape[1]):
            if ilcm[i,j] > sblk[i,j]:
                target_list.append([i,j])
                #print("target position:%d\t%d" % (i, j))
    return np.mat(target_list)

def get_th(ilcm):
    mu = np.mean(ilcm)
    sigma = np.std(ilcm)
    k = 5
    return mu + sigma*k

def get_target_pos(target_list):
    pos = target_list * step_size
    return pos

def draw_rectangle(pos, img, file_name):
    image = img.copy()
    new_pos = list()
    rec_size = 15
    print(pos)
    print(len(pos))
    if len(pos):
        x,y = [0,0]
        for i in range(len(pos)):
            x += pos[i,0]
            y += pos[i,1]
        x = int(x/len(pos))
        y = int(y/len(pos))
        new_pos = [x,y]
        cv2.rectangle(image, (new_pos[0]-rec_size,new_pos[1]+rec_size), (new_pos[0]+rec_size,new_pos[1]-rec_size), (0,255,0), 1)
        cv2.imshow('input image',img)
        cv2.imshow('output image',image)
        output_file = output_path+'.'+file_name
        print(output_file)
        #cv2.imwrite(output_file, image)
    else:
        print("no object detected!")

    return

#def draw_rectangle(pos, img):
#    image = img.copy()
#    new_pos = list()
#    rec_size = 15
#    pos = pos.tolist()
#    if len(pos) >= 1:
#        for i in range(len(pos)-1):
#            compare_pos.append([pos[0][0],pos[0][1]])
#            if (abs(pos[i][0]-pos[i][0])<sub_size or abs(pos[i][1]-pos[i][1])<sub_size):
#                new_x = int((pos[i][0]+pos[i+1][0])/2)
#                new_y = int((pos[i][1]+pos[i+1][1])/2)
#                new_pos.append([new_x, new_y])
#            else:
#                new_pos.append([pos[i,0], pos[i,1]])
#    else:
#        print("no object detected!")
#    print(pos)
#    print(new_pos)
#    for i in range(len(new_pos)):
#        cv2.rectangle(image, (new_pos[i][0]-rec_size,new_pos[i][1]+rec_size), (new_pos[i][0]+rec_size,new_pos[i][1]-rec_size), (0,255,0), 1)
#        cv2.imshow('input image',img)
#        cv2.imshow('output image',image)
#    #cv2.imwrite('001_new.jpg', img)
#    return

def img_process(input_path, output_path):
    #input_path ='./data_set/'
    #file_name = '1.bmp'
    file_name = input_path.split(".")[-1]
    print('loading %s' % input_path)
    img = cv2.imread(input_path)
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_mat = np.mat(img_grey)

    sblk , ln = get_sblk(img_mat, sub_size)
    ilcm = get_ilcm(ln, sblk)
    th = get_th(ilcm)
    target_list = get_target_th(ilcm, th)
    pos = get_target_pos(target_list)

    draw_rectangle(pos, img, file_name)
    
    return


output_path = '\\output\\'
img_path = gb.glob(".\data_set\*.bmp") 
sub_size = 8
step_size = int(sub_size/2)
img_process('./data_set/14.bmp',output_path)
#for path in img_path:
#    print(path)
#    #img  = cv2.imread(path) 
#    #print(path)
#    img_process(path, output_path)
#    pass
    

cv2.waitKey()
cv2.destroyAllWindows()

#if __name__ == '__main__':
#    main()