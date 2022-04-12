import os
import numpy as np
import tensorflow as tf

import time
from glob import glob
from PIL import Image
from skimage import io
from skimage.transform import resize
from tensorflow.python.ops.image_ops_impl import ssim

from tqdm import tqdm

import model
import utils

AUTO = tf.data.AUTOTUNE

# Hyper parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
NUM_EPOCHS = 500

IM_HEIGHT = 64
IM_WIDTH = 128

DATASET_DIR = "/media/shin/2nd_m.2/learningHDR_data/synthetic_data"
ENCODING_STYLE = "utf-8"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_RENDER_DIR = os.path.join(DATASET_DIR, "train_render")
VAL_DIR = os.path.join(DATASET_DIR, "test")
VAL_RENDER_DIR = os.path.join(DATASET_DIR, "test_render")

def setData(fname=''):
    # load ldr image, resize it to feed the our model
    im = resize(io.imread(fname), [IM_HEIGHT, IM_WIDTH]).astype('float32')
    ims = np.repeat(np.reshape(im, [1, IM_HEIGHT, IM_WIDTH, 3]), BATCH_SIZE, 0)
    return ims

def _read_ldr(ldr,hdr,rndr):
    ldr_image = np.array(Image.open(ldr.numpy()))
    ldr_image = tf.divide(tf.cast(ldr_image ,tf.float32), 255.)
    hdr_image = utils.openexr2np(hdr.numpy())
    rndr_image = utils.openexr2np(rndr.numpy())

    return ldr_image, hdr_image, rndr_image

def preprocessing(ldr, hdr, rndr):
    
    ldr_img, hdr_img, rndr_img = tf.py_function(_read_ldr,[ldr, hdr, rndr],[tf.float32,tf.float32, tf.float32])
    
    return ldr_img, hdr_img, rndr_img

def configureDataset(dirpath, train= "train"):
    
    ldr_path = str(dirpath + '/*.jpg')
    hdr_path = str(dirpath + '/*.exr')
    ldr_list = glob(ldr_path)
    hdr_list = glob(hdr_path)
    ldr_list.sort()
    hdr_list.sort()

    if train == True:
        render_path = str(TRAIN_RENDER_DIR + '/*.exr')
    else:
        render_path = str(VAL_RENDER_DIR + '/*.exr')
    
    render_list = glob(render_path)
    render_list.sort()
    
    ds = tf.data.Dataset.from_tensor_slices((ldr_list,hdr_list,render_list))
    ds = ds.map(preprocessing, num_parallel_calls=AUTO)    
    
    if train == True:
        ds = ds.shuffle(buffer_size=10000)
    
    ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    return ds

if __name__=="__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    """Path for tf.summary.FileWriter and to store model checkpoints"""
    root_dir=os.getcwd()
    filewriter_path = utils.createNewDir(root_dir, "tensorboard")
    root_logdir = os.path.join(filewriter_path, "logs/fit")
    logdir = utils.createNewDir(root_logdir)
    train_logdir, val_logdir = utils.createTrainValidationDirpath(logdir, createDir=False)
    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)
    
    print('tensorboard --logdir={}'.format(logdir))

    """Init Dataset"""
    train_ds = configureDataset(TRAIN_DIR, train=True)
    val_ds = configureDataset(VAL_DIR, train=False)

    """"Create Output Image Directory"""
    outputDir = utils.createNewDir(root_dir, "outputImg")
    outImgDir_timestamp= utils.createNewDir(outputDir, name=None)
    train_outImgDir, val_outImgDir = utils.createTrainValidationDirpath(outImgDir_timestamp, createDir=True)

    """"Create Output Render scene Directory"""
    outputRndr = utils.createNewDir(root_dir, "outputRenderScene")
    outputRndr_timestamp= utils.createNewDir(outputRndr, name=None)
    train_outputRndr, val_outputRndr = utils.createTrainValidationDirpath(outputRndr_timestamp, createDir=True)

    # import optimizer
    # learning_rate_fn = optimizer.AlexNetLRSchedule(initial_learning_rate = LEARNING_RATE, name="performance_lr")
    # _optimizer = optimizer.AlexSGD(learning_rate=learning_rate_fn, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, name="alexnetOp")
    _optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)

    """
     [deconv_method] 
        'upsample' -> nearest neighbor method, occurs checkerboard effect
        'resize' -> reduce checkerboard effect
    """
    _model = model.ldr2hdr(fc_dim=64, im_height=64, deconv_method='resize') 
    
    """
    Check dataset that properly work...
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    for i, (_, image, _) in enumerate(val_ds.take(5)):
        ax = plt.subplot(5,5,i+1)
        plt.imshow(image[i])
        plt.axis('off')
    plt.show()

    """CheckPoint Create"""
    # checkpoint_path = utils.createNewDir(root_dir, "checkpoints")
    # ckpt = tf.train.Checkpoint(
    #                         epoch = tf.Variable(1),
    #                         net=_model,
    #                        optimizer=_optimizer,
    #                        train_iterator=iter(train_ds),
    #                        val_intrator=iter(val_ds),)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    # print('Latest checkpoint restored!!')
    """"""

    with tf.device('/GPU:0'):
        @tf.function
        def train_step(ldrs, hdrs, rndrs):

            with tf.GradientTape() as tape:

                latent_vec = _model.encoder(ldrs, training = True)

                outImg, outRndr = _model.decoder(training = True)

                hdrs = tf.divide(tf.pow(hdrs, 1/2.2), 30.)

                l1_loss = tf.reduce_mean(tf.abs(outImg - hdrs))
                ssim_loss = tf.image.ssim(outImg, hdrs, max_val=1.)
                rndr_loss = tf.reduce_mean(tf.square(outRndr - rndrs))
                combine_loss = l1_loss + rndr_loss + ssim_loss

            gradients = tape.gradient(combine_loss, _model.trainable_variables)
            _optimizer.apply_gradients(zip(gradients, _model.trainable_variables))
            train_loss(combine_loss)

            return outImg, outRndr

        @tf.function
        def test_step(test_ldrs, test_hdrs, test_rndrs):
            latent_vec_test = _model.encoder(test_ldrs, training =False)
            outImg_test, outRndr_test = _model.decoder(training = False)
            
            test_hdrs = tf.divide(tf.pow(test_hdrs, 1/2.2), 30.)

            test_l1_loss = tf.reduce_mean(tf.abs(outImg_test- test_hdrs))
            test_rndr_loss = tf.reduce_mean(tf.square(outRndr_test - test_rndrs))
            ssim_loss = tf.image.ssim(outImg_test, test_hdrs, max_val=1.)
            test_combine_loss = test_l1_loss + test_rndr_loss + ssim_loss
            test_loss(test_combine_loss)

            return outImg_test, outRndr_test

    
    print("시작")
    for epoch in range(NUM_EPOCHS):

        start = time.perf_counter()

        train_loss.reset_states()
        test_loss.reset_states()
        
        for step, (ldrs, hdrs, rndrs) in enumerate(tqdm(train_ds)):
            with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
                outImg, outRndr = train_step(ldrs, hdrs, rndrs)
                if step % 10 == 0:
                    fullHDR = tf.pow(tf.math.multiply(30., outImg[0]), 2.2)
                    utils.writeHDR(fullHDR.numpy(), "{}/train_{}_in_epoch{}.exr".format(train_outImgDir, step,epoch), outImg.get_shape()[1:3])
                    utils.writeHDR(outRndr[0].numpy(), "{}/train_{}_in_epoch{}.exr".format(train_outputRndr, step,epoch), outRndr.get_shape()[1:3])

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
            
        for step, (ldrs, hdrs, rndrs) in enumerate(tqdm(val_ds)):
            outImg, outRndr = test_step(ldrs, hdrs, rndrs)
            if step % 10 == 0:
                fullHDR = tf.math.pow(tf.math.multiply(30., outImg[0]), 2.2)
                utils.writeHDR(fullHDR.numpy(), "{}/val_{}_in_epoch{}.exr".format(val_outImgDir,step,epoch), outImg.get_shape()[1:3])
                utils.writeHDR(outRndr[0].numpy(), "{}/val_{}_in_epoch{}.exr".format(val_outputRndr, step,epoch), outRndr.get_shape()[1:3])
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch+1)

        # ckpt.epoch.assign_add(1)
        # if int(ckpt.epoch) % 50 == 0:
        #     save_path =  ckpt_manager.save()
        #     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        
        print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss.result(), test_loss.result()))
        
        print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))

    print("끝")