from Models import  ResNet18, OlzhasNet45, MobileNetV2
import torch
from torch.autograd import Variable
import os
import time
import cv2
import numpy as np
import argparse
from Detector.core.detect import create_mtcnn_net, MtcnnDetector
from torchvision.transforms import ToTensor

def predict(model_name, model_path, images_path, vis = True):

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./Detector/model_store/pnet_epoch.pt",
                                        r_model_path="./Detector/model_store/rnet_epoch.pt",
                                        o_model_path="./Detector/model_store/onet_epoch.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)


    if model_name == 'ResNet':
        net = ResNet18()
    elif model_name == 'MobileNet':
        net = MobileNetV2()
    else:
        print("No such model")
        quit()

    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)

    net.eval()
    net.cuda()

    time_ = []
    with torch.no_grad():
        for subdir, dirs, files in os.walk(images_path):
            for file in files:
                  if file.endswith('.jpg'):

                    t = time.time()
                    text = 'No glasses!'

                    img = cv2.imread(os.sep.join([subdir, file]))
                    bboxs, landmarks = mtcnn_detector.detect_face(img, None)

                    if len(bboxs)==0:
                        print("Нет лиц на картинке!")
                        if vis:
                            cv2.putText(img, 'No face :(', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.imshow('Result', img)
                            k = cv2.waitKey(0)
                            if k==27:
                                quit()
                        continue

                    bbox = bboxs[0][0:4].astype(int)
                    for en, pt in enumerate(bbox):
                        if pt < 0:
                            bbox[en] = 0

                    x_left, y_top, x_right, y_bottom = bbox

                    cropped = img[y_top:y_bottom, x_left:x_right]
                    cropped = cv2.resize(cropped, (60, 60))
                    cropped = ToTensor()(cropped)

                    face_img = Variable(cropped.unsqueeze(0)).float()
                    face_img = face_img.cuda()
                    prob = net(face_img)


                    print("Вероятность очков", prob.item())
                    calc_time = time.time() - t
                    print(text)
                    print("Время вычесления одной картинки", calc_time)
                    time_.append(calc_time)

                    if prob > 0.2:
                        text = 'There are glasses!'

                    if vis:
                        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Result', img)
                        k = cv2.waitKey(0)
                        if k == 27:
                            quit()


    print("Среднее время обработки одной картинки", np.mean(np.array(time_)))

def parse_args():

  parser = argparse.ArgumentParser(formatter_class=
                                   argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model_name', default='ResNet',
                      help='Possible options: ResNet, MobileNet')
  parser.add_argument('--model_path', default='/home/oljike/PycharmProjects/GlassesClassification/weights/ResNet/ResNet.pt',
                      help='Path to trained model')
  parser.add_argument('--test_dir', default='./Dataset/test',
                      help='Path to directiry which contains test images')
  parser.add_argument('--vis', default=True,
                      help='If you want to see the results of algorithm visually')

  return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    predict(args.model_name, args.model_path, args.test_dir, args.vis)


