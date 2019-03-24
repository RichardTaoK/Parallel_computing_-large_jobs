"""Demo for use yolo v3
"""
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）


def process_image(img):
    image = cv2.resize(img, (416, 416),interpolation=cv2.INTER_CUBIC)##调整缩小和展开图像
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image  ##返回经过处理的图像（64,64,3）


def get_classes(file):##获取类名，参数是数据库的类名，返回列表和类名
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):##在图像上画框（原始图像，盒子对象，对象的分数，对象的类，所有的类名）
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        ##获取矩形点
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)##画矩形
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):##使用YOLO v3检测图像（参数是：原始图像，YOLO之前训练好的模型，所有的类名）
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)##用yolo对image进行分类，得分，画框
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))##总共使用时间

    if boxes is not None:##就说明有对象，可以进行画框
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_video(video, yolo, all_classes):
    """用yolo v3检测视频
    # 参数:
        video: 视=视频文件名.
        yolo: yolo训练模型.
        all_classes: 所有类名.
    """
    video_path = os.path.join("videos", "test", video)##video路径
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)##识别

    # 准备保存检测到的video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()
        if not res:
            break
        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)
        # 一帧一帧的保存图像
        vout.write(image)
        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()
    

if __name__ == '__main__':
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)

    # 在特测试文件夹中检测video
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            for f in files:
                print(f)
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = detect_image(image, yolo, all_classes)
                cv2.imwrite('images/res/' + f, image)

    video = 'library1.mp4'# 视频名
    detect_video(video, yolo, all_classes)
