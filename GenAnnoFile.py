import os
import random
import Augmentor
import cv2
from Detector.core.detect import create_mtcnn_net, MtcnnDetector
import argparse

path_to_selfie_txt = '/home/oljike/PycharmProjects/GlassDataset/Dataset/negative/selfie_dataset.txt'
dir_to_negative = '/home/oljike/PycharmProjects/GlassDataset/Dataset/negative/images'
dir_to_positive = '/home/oljike/PycharmProjects/GlassDataset/Dataset/positive'
anno_dir = 'annotations/'


def crop_faces(path_to_dataset, is_positive):
    save_dir = os.path.join(path_to_dataset, 'cropped')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if is_positive:
        images_dir = os.path.join(path_to_dataset, 'positive')
        save_dir = os.path.join(save_dir, 'positive')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        images_dir = os.path.join(path_to_dataset, 'negative')
        save_dir = os.path.join(save_dir, 'negative')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./Detector/model_store/pnet_epoch.pt",
                                        r_model_path="./Detector/model_store/rnet_epoch.pt",
                                        o_model_path="./Detector/model_store/onet_epoch.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    cnt = 0
    for subdir, dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.sep.join([subdir, file]))
                if img is not None:
                    bboxs, _ = mtcnn_detector.detect_face(img, os.sep.join([subdir, file]))
                    if bboxs is not None and len(bboxs) > 0:
                        if cnt % 100 == 0:
                            print("Number of preprocessed images", cnt)
                        cnt += 1
                        bbox = bboxs[0][0:4].astype(int)
                        for en, pt in enumerate(bbox):
                            if pt < 0:
                                bbox[en] = 0
                        x_left, y_top, x_right, y_bottom = bbox

                        cropped = img[y_top:y_bottom, x_left:x_right]
                        save_path = os.path.join(save_dir, 'pos_' + str(cnt) + '.jpg')
                        cv2.imwrite(save_path, cropped)


def get_data(dir_to_dataset, anno_dir):
    pos_data_path = os.path.join(dir_to_dataset, 'positive')
    pos_label = str(1)

    neg_data_path = os.path.join(dir_to_dataset, 'negative')
    neg_label = str(0)

    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    anno = open(os.path.join(anno_dir, 'anno_all.txt'), 'a')

    cnt = 0
    for subdir, dirs, files in os.walk(pos_data_path):
        for file in files:
            if file.endswith('.jpg'):
                anno.write(os.sep.join([subdir, file]) + ' ' + pos_label + '\n')
                cnt += 1

    for subdir, dirs, files in os.walk(neg_data_path):
        for file in files:
            if file.endswith('.jpg'):
                anno.write(os.sep.join([subdir, file]) + ' ' + neg_label + '\n')
                cnt += 1

    print("Всего %d фотографий людей!" % cnt)
    anno.close()


def split_data(anno_dir):
    lines = open(os.path.join(anno_dir, 'anno_all.txt')).readlines()
    random.shuffle(lines)
    open(os.path.join(anno_dir, 'anno_train.txt'), 'w').writelines(lines[:int(0.9 * len(lines))])
    open(os.path.join(anno_dir, 'anno_test.txt'), 'w').writelines(lines[int(0.9 * len(lines)):])


def augment_positive_data(dir_path):
    cnt = 0
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg'):
                cnt += 1

    p = Augmentor.Pipeline(dir_path)
    p.crop_random(probability=0.5, percentage_area=0.7)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.flip_left_right(probability=0.5)

    p.sample(int(cnt * 0.8))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='./Dataset/cropped/',
                        help='Path to directory with cropped images ')
    parser.add_argument('--anno_dir', default='./annotations',
                        help='Path to directiry which contains annotation files')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # crop_faces('/home/oljike/PycharmProjects/GlassDataset/Dataset', True)
    # crop_faces('/home/oljike/PycharmProjects/GlassDataset/Dataset', False)

    augment_positive_data(os.path.join(args.data_path, 'positive'))
    get_data(args.data_path, args.anno_dir)
    split_data(args.anno_dir)
