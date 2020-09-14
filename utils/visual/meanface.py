import numpy as np
import pandas as pd
import cv2
from PIL import Image
import shutil
import os

# calculate mean_face
def mean_face(base_dir, save_dir):
    count = 0
    total = np.zeros((256, 256,3), dtype = np.float64)
    for face_path in os.listdir(base_dir):
        face_path_dir = os.path.join(base_dir, face_path)
        img = cv2.imread(face_path_dir).astype(np.float64)
        total += img
        count += 1
    av_image = (total / float(count)).astype(np.uint8)
    cv2.imwrite(save_dir+'av_image.jpg', av_image)


# risk feature face
def risk_face(nosort_file, res):
    nosort_csv = pd.read_csv(nosort_file, encoding='gbk')
    res_csv = pd.read_csv(res, encoding='gbk')
    nosort_csv = pd.merge(nosort_csv, res_csv, on='face_picture', how='left')
    nosort_csv = nosort_csv[['face_picture', 'FACE_SCORE']]
    sort_csv = nosort_csv.sort_values('FACE_SCORE', ascending=True)
    # choose pre_100 pictures path
    sort_csv_former = sort_csv.iloc[0:100, :].iloc[:, 0].values
    sort_csv_later = sort_csv.iloc[-101:-1, :].iloc[:, 0].values

    return sort_csv_former, sort_csv_later


# merge the risk_pictures
def merge_risk_picture(img_dir):
    img1, img2, img3, img4, img5, img6, img7, img8, img9, img10 = [],[],[],[],[],[],[],[],[],[]
    for i, image_name in enumerate(os.listdir(img_dir)):
        image_path = os.path.join(img_dir, image_name)
        img = cv2.imread(image_path)
        if i<10:
            img1.append(img)
        elif 20 > i >= 10:
            img2.append(img)
        elif 30 > i >= 20:
            img3.append(img)
        elif 40 > i >= 30:
            img4.append(img)
        elif 50 > i >= 40:
            img5.append(img)
        elif 60 > i >= 50:
            img6.append(img)
        elif 70 > i >= 60:
            img7.append(img)
        elif 80 > i >= 70:
            img8.append(img)
        elif 90 > i >= 80:
            img9.append(img)
        else:
            img10.append(img)
    # merge the risk faces
    vtitch = np.vstack((np.hstack(tuple(img1)), np.hstack(tuple(img2)), np.hstack(tuple(img3)), np.hstack(tuple(img4)), np.hstack(tuple(img5)),
                        np.hstack(tuple(img6)), np.hstack(tuple(img7)), np.hstack(tuple(img8)), np.hstack(tuple(img9)), np.hstack(tuple(img10))))
    cv2.imwrite(img_dir+'sort_face.jpg', vtitch)


if __name__ == '__main__':
    # base_dir, save_dir
    base_dir = './image/'
    save_dir = './log/'
    nosort_file = './'
    res = './checkpoints/csv/Res_facepicturename.csv'
    mean_face(base_dir, save_dir)
    sort_csv_former, sort_csv_later = risk_face(nosort_file, res)
    merge_risk_picture(save_dir)


