import os
import tensorflow as tf
import numpy as np
import loadSUN360 as lmd
from multiprocessing import Process
import math

# 이미지와 라벨로부터 tfrecord 생성하는 모듈 
# 독립적으로 실행되는 모듈
# Reference: https://www.kaggle.com/ryanholbrook/tfrecords-basics

TRAIN_IMAGE_DIR = r"D:\ILSVRC2012\ILSVRC2012_img_train"
TEST_IMAGE_DIR = r"D:\ILSVRC2012\ILSVRC2012_img_val"
TRAIN_TFREC_DIR = r"D:\ILSVRC2012\class10_q95_tfrecord_train"
TEST_TFREC_DIR = r"D:\ILSVRC2012\class10_q95_tfrecord_val"
TFRECORD_FILE_NAME= lambda name : '{}.tfrecord'.format(name)
IMAGE_SIZE = 256
IMAGE_ENCODING_QUALITY = 95  # default 95
TFRECORD_OPTION = tf.io.TFRecordOptions(compression_type="GZIP")

def _int64_feature(value):
  # if not isinstance(values, (tuple, list)):
  #   values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_ds(image, label):
  feature_description = {
    'image': _bytes_feature(image),
    'label': _int64_feature(label),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
  return example_proto.SerializeToString()

def convert_image_to_bytes(image="image"):
    #  이미지 받으면 RGB 순서로 받아짐. BGR로 받을지 안받을지는 알아서 결정해
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    height, width, _ = np.shape(image)  # return int
    
    # 한쪽만 256인 경우는 고려 안해줌. 256사이즈에 대해 그냥 resize하는 이유는 별로 시간 안걸리니까

    # GPU로 돌리니까 이상하게 파싱되는듯 ㅠㅠ 메인함수에서 풀때 참조를 못함... 절대 쓰지말자
    if width <= IMAGE_SIZE and height <= IMAGE_SIZE:
        cropped_img = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.BILINEAR)
    
    elif width > height:
        center = (width - IMAGE_SIZE) / 2
        start = math.floor(center)
        end = math.ceil(center)
        resized_img = tf.image.resize(image, [IMAGE_SIZE,width], method=tf.image.ResizeMethod.BILINEAR)
        cropped_img = resized_img[:,start:-end,:]
        
    else:
        center = (height - IMAGE_SIZE) / 2
        start = math.floor(center)
        end = math.ceil(center)
        resized_img = tf.image.resize(image, [height,IMAGE_SIZE], method=tf.image.ResizeMethod.BILINEAR)
        cropped_img = resized_img[start:-end,:,:]
    
    # cv2 방식
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_ENCODING_QUALITY]
    # _, im_buf_arr = cv2.imencode(".jpg", cropped_img, encode_param)
    # img_raw = im_buf_arr.tobytes()

    convert_image = tf.cast(cropped_img, tf.uint8)

    img_raw = tf.io.encode_jpeg(convert_image, quality=IMAGE_ENCODING_QUALITY)
    
    return img_raw

def parse_to_tfrecord(meta_data = "meta_data", splited_dir_list = "splited_dir_list", 
                        tfrecord_dir= "tfrecord_dir", train=None,  mdir = "mdir",
                          image_dir = "image_dir", mindex = "mindex"):

    for dir_path in splited_dir_list:
        
        index = mdir.index(dir_path)

        ###### class 10개 추려내는 작업
        index = mindex[index]
        
        if 440 <= index and index < 450:
        ######

          in_dir_path = os.path.join(image_dir, dir_path)

          new_dir = os.path.join(tfrecord_dir, dir_path)

          all_file_list = os.listdir(in_dir_path)

          if not os.path.isdir(new_dir):
            print("new_dir", new_dir)
            os.mkdir(new_dir)
          os.chdir(new_dir)

          
          for file_name in all_file_list:
              
              # if os.path.isfile(os.path.join(new_dir,TFRECORD_FILE_NAME(file_name.split(".")[0]))):
              #   # print("already exists file name",file_name)
              #   continue
              # else:
                file_path = os.path.join(in_dir_path, file_name)
                with open(file_path, 'rb') as f:
                    raw_image = f.read()
                    original_image = tf.io.decode_jpeg(raw_image)

                image_bytes = convert_image_to_bytes(original_image)

                if train:
                  with tf.io.TFRecordWriter(TFRECORD_FILE_NAME(file_name.split(".")[0]), TFRECORD_OPTION) as writer:
                    print("parsing train",file_name)
                    example = serialize_ds(image_bytes, index)
                    writer.write(example)
                  writer.close()
                else:
                  with tf.io.TFRecordWriter(TFRECORD_FILE_NAME(file_name.split(".")[0]), TFRECORD_OPTION) as writer:
                    print("parsing test",file_name)
                    example = serialize_ds(image_bytes, index)
                    writer.write(example)
                  writer.close()
        # else:
        #   print("already exists ", new_dir)    

    print("finish loading dataset of",image_dir)
    
if __name__ == "__main__":

  # dir, index, name
  metadata = lmd.load_ILSVRC2012_metadata()

  _dir, _index, _name = metadata

  #train
  # if not os.path.isdir(TRAIN_TFREC_DIR):
  #     os.mkdir(TRAIN_TFREC_DIR)
  # os.chdir(TRAIN_TFREC_DIR)
  # train_dir_list = os.listdir(TRAIN_IMAGE_DIR)
  # print("train_dir_list",len(train_dir_list))
  # # parse_to_tfrecord(meta_data = metadata, splited_dir_list = train_dir_list, 
  # #                       tfrecord_dir= TRAIN_TFREC_DIR, train=True,  mdir = _dir,
  # #                         image_dir = TRAIN_IMAGE_DIR, mindex = _index)
  # split_number = math.ceil(len(train_dir_list) / 4)
  # train_splited_dir_list = [train_dir_list[x:x + split_number] for x in range(0, len(train_dir_list), split_number)]

  # p1 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[0], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p2 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[1], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p3 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[2], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p4 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[3], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p5 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[4], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p6 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[5], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p7 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[6], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p8 = Process(target=parse_to_tfrecord,
  #             args=(metadata, train_splited_dir_list[7], 
  #               TRAIN_TFREC_DIR, True, _dir,
  #                 TRAIN_IMAGE_DIR, _index))
  # p1.start()
  # p2.start()
  # p3.start()
  # p4.start()
  # p5.start()
  # p6.start()
  # p7.start()
  # p8.start()

  # p1.join()
  # p2.join()
  # p3.join()
  # p4.join()
  # p5.join()
  # p6.join()
  # p7.join()
  # p8.join()

  
  #test
  if not os.path.isdir(TEST_TFREC_DIR):
      os.mkdir(TEST_TFREC_DIR)
  os.chdir(TEST_TFREC_DIR)

  val_dir_list = os.listdir(TEST_IMAGE_DIR)
  
  split_number = math.ceil(len(val_dir_list) / 5)
  test_splited_dir_list = [val_dir_list[x:x + split_number] for x in range(0, len(val_dir_list), split_number)]

  tp1 = Process(target=parse_to_tfrecord,
              args=(metadata, test_splited_dir_list[0], 
                TEST_TFREC_DIR, False, _dir,
                  TEST_IMAGE_DIR, _index))
  tp2 = Process(target=parse_to_tfrecord,
              args=(metadata, test_splited_dir_list[1], 
                TEST_TFREC_DIR, False, _dir,
                  TEST_IMAGE_DIR, _index))
  tp3 = Process(target=parse_to_tfrecord,
              args=(metadata, test_splited_dir_list[2], 
                TEST_TFREC_DIR, False, _dir,
                  TEST_IMAGE_DIR, _index))
  tp4 = Process(target=parse_to_tfrecord,
              args=(metadata, test_splited_dir_list[3], 
                TEST_TFREC_DIR, False, _dir,
                  TEST_IMAGE_DIR, _index))
  # tp5 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[4], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))        
  # tp6 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[5], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # tp7 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[6], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))
  # tp8 = Process(target=parse_to_tfrecord,
  #             args=(metadata, test_splited_dir_list[7], 
  #               TEST_TFREC_DIR, False, _dir,
  #                 TEST_IMAGE_DIR, _index))                                
  tp1.start()
  tp2.start()
  tp3.start()
  tp4.start()
  # tp5.start()
  # tp6.start()
  # tp7.start()
  # tp8.start()
  