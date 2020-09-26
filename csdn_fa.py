import os
import cv2
import numpy
import logging
import tensorflow as tf

from mtcnn import MTCNN 
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s %(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
 
imgSize1 = [112,96]
imgSize2 = [112,112]
coord5point1 = [[30.2946, 51.6963],  # 112x96的目标点
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]
coord5point2 = [[30.2946+8.0000, 51.6963], # 112x112的目标点
               [65.5318+8.0000, 51.6963],
               [48.0252+8.0000, 71.7366],
               [33.5493+8.0000, 92.3655],
               [62.7299+8.0000, 92.3655]]
 
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])
 
def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst
 
def main():
    pic_path = './examples/test_HR/'
    minsize = 50  # minimum size of face
    threshold = [0.2, 0.3, 0.3]  # three steps's threshold
    factor = 0.709  # scale factor
    detector = MTCNN(steps_threshold=threshold)
 
    # Size Parameter
    lower_threshold = 100
    upper_threshold = 200
    num = 0
 
    pic_name_list = os.listdir(pic_path)
    for every_pic_name in pic_name_list:
        img_im = cv2.imread(pic_path + every_pic_name)
        json = detector.detect_faces(img_im)
        if img_im is None or len(json)==0:
            continue
        else:
            shape = img_im.shape
            height = shape[0]
            width = shape[1]
            #json = detector.detect_faces(img_im)
            print(json[0]['confidence'], json[0]['box'])
            bounding_boxes, keys = json[0]['box'], json[0]['keypoints']
            bounding_boxes = numpy.array([bounding_boxes])
            points= []
            points.append(keys['left_eye'][0])
            points.append(keys['right_eye'][0])
            points.append(keys['nose'][0])
            points.append(keys['mouth_left'][0])
            points.append(keys['mouth_right'][0])
            points.append(keys['left_eye'][1])
            points.append(keys['right_eye'][1])
            points.append(keys['nose'][1])
            points.append(keys['mouth_left'][1])
            points.append(keys['mouth_right'][1])
            #assert(0)
            if bounding_boxes.shape[0] > 0:
                for i in range(bounding_boxes.shape[0]):
                    x1, y1, x2, y2 = int(min(bounding_boxes[i][0], min(points[:5]))), \
                                     int(min(bounding_boxes[i][1], min(points[5:]))), \
                                     int(max(bounding_boxes[i][2], max(points[:5]))), \
                                     int(max(bounding_boxes[i][3], max(points[5:])))
                    #new_x1 = max(int(1.50 * x1 - 0.50 * x2),0)
                    #new_x2 = min(int(1.50 * x2 - 0.50 * x1),width-1)
                    #new_y1 = max(int(1.50 * y1 - 0.50 * y2),0)
                    #new_y2 = min(int(1.50 * y2 - 0.50 * y1),height-1)
                    new_x1 = max(int(1.30 * x1 - 0.30 * x2),0)
                    new_x2 = min(int(1.30 * x2 - 0.30 * x1),width-1)
                    new_y1 = max(int(1.30 * y1 - 0.30 * y2),0)
                    new_y2 = min(int(1.30 * y2 - 0.30 * y1),height-1)
 
                    left_eye_x = points[:5][0]
                    right_eye_x = points[:5][1]
                    nose_x = points[:5][2]
                    left_mouth_x = points[:5][3]
                    right_mouth_x = points[:5][4]
                    left_eye_y = points[5:][0]
                    right_eye_y = points[5:][1]
                    nose_y = points[5:][2]
                    left_mouth_y = points[5:][3]
                    right_mouth_y = points[5:][4]
 
                    new_left_eye_x = left_eye_x - new_x1
                    new_right_eye_x = right_eye_x - new_x1
                    new_nose_x = nose_x - new_x1
                    new_left_mouth_x = left_mouth_x - new_x1
                    new_right_mouth_x = right_mouth_x - new_x1
                    new_left_eye_y = left_eye_y - new_y1
                    new_right_eye_y = right_eye_y - new_y1
                    new_nose_y = nose_y - new_y1
                    new_left_mouth_y = left_mouth_y - new_y1
                    new_right_mouth_y = right_mouth_y - new_y1
 
                    face_landmarks = [[new_left_eye_x,new_left_eye_y], # 在扩大100%人脸图中关键点坐标
                                      [new_right_eye_x,new_right_eye_y],
                                      [new_nose_x,new_nose_y],
                                      [new_left_mouth_x,new_left_mouth_y],
                                      [new_right_mouth_x,new_right_mouth_y]]
                    face = img_im[new_y1: new_y2, new_x1: new_x2] # 扩大100%的人脸区域
                    #dst1 = warp_im(face,face_landmarks,coord5point1) # 112x96对齐后尺寸
                    dst2 = warp_im(face,face_landmarks,coord5point2) # 112x112对齐后尺寸
                    #crop_im1 = dst1[0:imgSize1[0],0:imgSize1[1]]
                    crop_im2 = dst2[0:imgSize2[0],0:imgSize2[1]]
                    #cv2.imwrite(pic_path + every_pic_name[:-4] + '_' + str(num) + '_align_112x96.jpg',crop_im1)
                    cv2.imwrite(pic_path + every_pic_name[:-4] + '_' + str(num) + '_align_112x112.jpg',crop_im2)
                    num = num + 1
 
if __name__ == '__main__':
    main()
    cv2.waitKey()
    pass
