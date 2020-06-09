import os
import cv2
import random

path_trg = './data/RaFD/preprocessed'
path_src = './data/RaFD/data'

def process_image(img_path):
    rgb_img = cv2.imread(img_path)
    if rgb_img is None:
        print("Unable to load image: {}".format(img_path))
        return None

    cropped_img = rgb_img[170:770, 40:640]
    resized_img = cv2.resize(cropped_img, (256, 256))
    if resized_img is None:
        print("Unable to pre-process image: {}".format(img_path))
        return None
    return resized_img


def main():
    if not os.path.isdir(path_trg):
        os.mkdir(path_trg)

    id_dict = {}
    for root, dirs, files in os.walk(path_src):
        for f in files:
            identity = f.split('_')[1]
            if identity not in id_dict:
                id_dict[identity] = []
            id_dict[identity].append(f)

    ids = list(id_dict.keys())
    random.seed(1234)
    random.shuffle(ids)

    train_dir = os.path.join(path_trg, 'train')
    test_dir = os.path.join(path_trg, 'test')
    for d in [train_dir, test_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)

    for i, identity in enumerate(ids):
        mode = 'train' if i < 60 else 'test'
        for f in id_dict[identity]:
            emotion = f.split('_')[4]
            emotion_dir = os.path.join(path_trg, mode, emotion)
            if not os.path.isdir(emotion_dir):
                os.mkdir(emotion_dir)
            preprocessed_img = process_image(os.path.join(path_src, f))
            cv2.imwrite(os.path.join(emotion_dir, f), preprocessed_img)
        print('{} identities processed.'.format(i+1))

if __name__ == '__main__':
    main()
