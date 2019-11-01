import os
import sys
import xml.etree.ElementTree as ET
import glob
import os.path as osp


def xml_to_txt(indir, outdir):
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    # print(annotations)
    for i, file in enumerate(annotations):

        file_save = file.split('.')[0] + '.txt'
        # file_txt=os.path.join(outdir,file_save)
        file_txt = outdir + '/'+file_save

        # print(file_save)
        # os.makedirs(file_txt, exist_ok=True)
        f_w = open(file_txt, 'w')

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename').text  # 这里是xml的根，获取filename那一栏
        for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text  # 这里获取多个框的名字，底下是获取每个框的位置
            if name == 'crazing':
                index = 0
            elif name == 'inclusion':
                index = 1
            elif name == 'patches':
                index = 2
            elif name == 'pitted_surface':
                index = 3
            elif name == 'rolled-in_scale':
                index = 4
            elif name == 'scratches':
                index = 5
            xmlbox = obj.find('bndbox')
            xn = xmlbox.find('xmin').text
            xx = xmlbox.find('xmax').text
            yn = xmlbox.find('ymin').text
            yx = xmlbox.find('ymax').text
            # print xn
            # f_w.write(filename + ' ' + xn + ' ' + yn + ' ' + xx + ' ' + yx + ' ')
            # f_w.write(name + '\n')
            f_w.write(str(index) + ' ' + xn + ' ' + yn + ' ' + xx + ' ' + yx + ' ')
            f_w.write('\n')
    print('tranction finished.')

if __name__ == "__main__":

    indir = 'C:/Users/11855/Desktop/PyTorch-YOLOv3-master/PyTorch-YOLOv3-master/data/NEU-DET/valid/label'  # xml目录
    outdir = 'C:/Users/11855/Desktop/PyTorch-YOLOv3-master/PyTorch-YOLOv3-master/data/NEU-DET/valid/labels'  # txt目录

    xml_to_txt(indir, outdir)

    # a=sorted(glob.glob("%s/*.*" % indir))
    # print(len(a))
    # for i in a:
    #     if i.split('/')[-1].split('.')[-1]=='txt':
    #         os.remove(i)
