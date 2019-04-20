import dlib
import numpy as np
import cv2
import os

def detection(image_dir, save_dir):
    detector = dlib.get_frontal_face_detector()
    # read the face picture
    for image_path in os.listdir(image_dir):
        image_path_dir = os.path.join(image_dir, image_path)
        img = cv2.imdecode(np.fromfile(image_path_dir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # read the width and height
        img_shape = img.shape
        img_height = img_shape[0]
        img_width = img_shape[1]

        # dlib detection
        dets = detector(img, 1)
        print("face numbers:", len(dets))

        area_list, height_list, width_list = [], [], []
        for k, d in enumerate(dets):
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # calculate matrix area
            height = d.bottom() - d.top()
            width = d.right() - d.left()
            area = height*width
            area_list.append(area)
            height_list.append(height)
            width_list.append(width)
        # choose max_area det
        max_value = max(area_list)
        max_index = area_list.index(max_value)
        dets = dets[max_index]
        height = height_list[max_index]
        width = width_list[max_index]


        # set blank picture
        img_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            if d.top()+i >= img_height: # Prevent cross-border
                continue
            for j in range(width):
                if d.left()+j >= img_width:
                    continue
                img_blank[i][j] = img[d.top()+i][d.left()+j]

        cv2.imencode('.jpg', img_blank)[1].tofile(os.path.join(save_dir,image_path))


if __name__ == '__main__':
    image_dir = './dataset/image/'
    save_dir = './record/align/'
    detection(image_dir, save_dir)


