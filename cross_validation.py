from glob import glob
from pathlib import Path
import os
import shutil
import albumentations as A
import cv2
from contextlib import contextmanager


class Transform:
    horizontal_flip_transform = A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    vertical_flip_transform   = A.Compose([A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def create_label(image_path, label_path):
        with open(label_path, 'r') as f:
            label, class_label = [], []
            for i in f.readlines():
                i = i.split(' ')
                class_label.append(int(i[0]))
                label.append(list(map(float, i[1:])))
        return cv2.imread(image_path), label, class_label, os.path.splitext(os.path.basename(image_path))[0]

    def create_label_file(transformed, filename, save_dir):
        image = transformed['image']
        label = transformed['bboxes']
        class_label = transformed['class_labels']

        cv2.imwrite(save_dir / 'images' / (filename + '.jpg'), image)
        with open(save_dir / 'labels' / (filename + '.txt'), 'w') as f:
            f.write('\n'.join(list(map(lambda x: ' '.join([str(int(x[0]))] + list(map(str, x[1]))), zip(class_label, label)))))

    def identity(image, label, save_dir):
        shutil.copy(image, save_dir / 'images' / Path(image).name)
        shutil.copy(label, save_dir / 'labels' / Path(label).name)

    def horizontal_flip(image, label, save_dir):
        image, label, class_label, filename = Transform.create_label(image, label)
        transformed = Transform.horizontal_flip_transform(image=image, bboxes=label, class_labels=class_label)
        Transform.create_label_file(transformed, filename + '_HorizontalFlip', save_dir)

    def vertical_flip(image, label, save_dir):
        image, label, class_label, filename = Transform.create_label(image, label)
        transformed = Transform.vertical_flip_transform(image=image, bboxes=label, class_labels=class_label)
        Transform.create_label_file(transformed, filename + '_VerticalFlip', save_dir)


class CrossValidationDataset:
    def __init__(self, path, train, valid, test, augment=['identity']):
        self.path  = Path(path)
        self.train = train
        self.valid = valid
        self.test  = test
        self.augment  = augment
        self.num   = 0
        self.sum   = self.train + self.valid + self.test
        self.enume = list(map(str, range(self.sum)))

        self.temp_file_data = {}
        if os.path.isfile(os.path.join(self.path, '.cross_validation')):
            with open(os.path.join(self.path, '.cross_validation'), 'r') as f:
                for f_ in f.read().split('\n'):
                    f_ = f_.split('::')
                    self.temp_file_data[f_[0]] = f_[1:]

    def __iter__(self):
        return self

    def __next__(self):
        if self.num >= self.sum:
            self.num   = 0
            self.enume = list(map(str, range(self.sum)))
            with open(os.path.join(self.path, '.cross_validation'), 'w') as f:
                f.write('\n'.join(map(lambda x: '::'.join([x[0]] + x[1]), self.temp_file_data.items())))
            raise StopIteration()
        
        if self.num != 0:
            self.enume = self.enume[1:] + [self.enume[0]]
        self.num += 1

        if str(self.enume) in self.temp_file_data.keys():
            return self.temp_file_data[str(self.enume)]

        train = self.enume[:self.train]
        valid = self.enume[self.train:self.train+self.valid]
        test  = self.enume[self.train+self.valid:]

        train_data = []
        for i in train:
            train_data += glob(str(self.path / i / 'images' / '*.jpg'))

        valid_data = []
        for i in valid:
            valid_data += glob(str(self.path / i / 'images' / '*.jpg'))

        test_data = []
        for i in test:
            test_data += glob(str(self.path / i / 'images' / '*.jpg'))

        save_dir = 'temp'
        # save_dir = self.path / (save_dir + str(0)) # delete
        for i in range(1000):
            if not os.path.isdir(str(self.path / (save_dir + str(i)))):
                save_dir = self.path / (save_dir + str(i))
                os.mkdir(str(save_dir))
                for name in ('train', 'valid', 'test'):
                    os.mkdir(str(save_dir / name))
                    os.mkdir(str(save_dir / name / 'images'))
                    os.mkdir(str(save_dir / name / 'labels'))
                break
        
        shutil.copy(self.path / 'data.yaml', save_dir / 'data.yaml')
        shutil.copy(self.path / 'data.yaml', save_dir / 'test.yaml')
        with open(save_dir / 'data.yaml', 'r') as f:
            data_string = f.readlines()
        data_string[0] = 'train: {}\n'.format(save_dir / 'train' / 'images')
        data_string[1] = 'val: {}\n'.format(save_dir / 'valid' / 'images')
        data_string[2] = 'test: {}\n'.format(save_dir / 'valid' / 'images')
        with open(save_dir / 'data.yaml', 'w') as f:
            f.write(''.join(data_string))
        data_string[0] = 'train: {}\n'.format(save_dir / 'train' / 'images')
        data_string[1] = 'val: {}\n'.format(save_dir / 'test' / 'images')
        data_string[2] = 'test: {}\n'.format(save_dir / 'test' / 'images')
        with open(save_dir / 'test.yaml', 'w') as f:
            f.write(''.join(data_string))


        save_train_dir = save_dir / 'train'
        for img in train_data:
            for arg in self.augment:
                lbl = img.split('\\')
                lbl[-2] = 'labels'
                lbl[-1] = lbl[-1].replace('.jpg', '.txt')
                lbl = '\\'.join(lbl)
                getattr(Transform, arg)(img, lbl, save_train_dir)
        
        valid_image_temp_dir = save_dir / 'valid' / 'images'
        valid_label_temp_dir = save_dir / 'valid' / 'labels'
        for data in valid_data:
            shutil.copy(data, valid_image_temp_dir / Path(data).name)
            data = data.split('\\')
            data[-2] = 'labels'
            data[-1] = data[-1].replace('.jpg', '.txt')
            data = '\\'.join(data)
            shutil.copy(data, valid_label_temp_dir / Path(data).name)

        test_image_temp_dir = save_dir / 'test' / 'images'
        test_label_temp_dir = save_dir / 'test' / 'labels'
        for data in test_data:
            shutil.copy(data, test_image_temp_dir / Path(data).name)
            data = data.split('\\')
            data[-2] = 'labels'
            data[-1] = data[-1].replace('.jpg', '.txt')
            data = '\\'.join(data)
            shutil.copy(data, test_label_temp_dir / Path(data).name)

        save_dir = str(save_dir)

        data_yaml = os.path.join(save_dir, 'data.yaml')
        test_yaml = os.path.join(save_dir, 'test.yaml')
        self.temp_file_data[str(self.enume)] = [data_yaml, test_yaml]

        return data_yaml, test_yaml


@contextmanager
def delete_temp_file(dir_path):
    yield dir_path
    shutil.rmtree(os.path.split(dir_path[0])[0])
