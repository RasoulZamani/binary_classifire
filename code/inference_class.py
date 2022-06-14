# importing necessary modules _________________________________________________________________

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# class definitaion____________________________________________________________________________
class InferenceClass():
    """
    this class get directory address of model and images list
    then load model and inference on input batch to predict class probabilities
    Args:
        model_dir: adsress of directory contains model
        
    it has a getScore function that can inference on images:
    Args: 
        imgs : list of rgb array images or directory contains images
    Returns:
        cls_pro: class probabilities (batch,2)
    """
    
    def __init__(self, model_dir):
        # load model
        self.model = tf.keras.models.load_model(model_dir)
    
    def getScore(self,imgs):
        """
        geting imags array list and calculate probabilities of classes for all images in batch
        for more generality if imgs is a directory,or one image address it converted to list of
        array images. also for pre-processing we apply :
                grayscale: (a,b,3) -> (a,b,1)
                rescale: [0,255] -> [0,1]
                reshape: (a,b) -> (32,32)
        in order to make image ready for inferencing by model.
        
        Args:
                imgs: list of image array (one image address or directory also is acceptable)
        Returns:
                class probabilities 
        
        """
        # first assume: imgs itself is list of array images __________________________________
        img_arr_list = imgs
        
        #anyway, if imgs is not a list of arr then we chek is it a directory or img address
        if (type(imgs)== str) :     # imgs is image adress or directory
            if os.path.isdir(imgs): # imgs is a directory:
                img_dir = imgs
                img_adrs_list = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
                
                img_arr_list=[]
                for img_adrs in img_adrs_list:
                    img = tf.keras.preprocessing.image.load_img(img_adrs)
                    img_arr = tf.keras.preprocessing.image.img_to_array(img)
                    img_arr_list.append(img_arr)
            
            else:                   # imgs is an image address
                image_adrs = imgs
                img = tf.keras.preprocessing.image.load_img(image_adrs)
                img_arr = tf.keras.preprocessing.image.img_to_array(img)
                img_arr_list=[img_arr]
                
        # pre process image and make it ready for prediction ________________________________
        
        pre_img_list = [] # list of pre processed array images 
        for img_arr in img_arr_list:
            
            img_gry = tf.image.rgb_to_grayscale(img_arr)
            
            # resizing image into input shape of model
            img_res = tf.image.resize(img_gry, size=(32,32))
        
            # rescale to [0,1]
            img_max = tf.math.reduce_max(img_res)
            img_pre = tf.keras.layers.Rescaling(1./img_max)(img_res)
        
            # adding new preprocessed image to new pre_img_list
            pre_img_list.append(img_pre)
            
        # convert list to array 
        pre_imgs = np.array(pre_img_list)
        
        # batch interface
        pred = self.model.predict_on_batch(pre_imgs)
        
        q_list=[] # list of other class probabilities
        for p in pred:
            q_list.append(1-p)
        q_arr = np.array(q_list)
        
        # probabilites of both class in array(batch,2) as task wanted 
        cls_prob = np.concatenate((pred, q_arr), axis=1)
        
        return cls_prob
    
# if this code run as a main script ______________________________________________________________
if __name__ == "__main__":
    """
    this part of code,  get model name(from models are availbel in model directory)
    and imge list or directory, then instantiate class, predict on samples and print
    probabilities.Also you can see time of loading and running
    
    if you have another model, assign it to model_dir in input of class
    if you have another images, assign it to imgs in getScore method 
    
    """
    # directory addresses
    CODE_DIR   = os.getcwd()
    ROOT_DIR   = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
    MODEL_DIR  = os.path.join(ROOT_DIR, 'model')
    MODEL_NAME = "model_50epocks"
    MODEL_PATH = os.path.join(MODEL_DIR,MODEL_NAME )
    IMG_DIR    = os.path.join(ROOT_DIR, r"internet_data_test\alp")
    SHOW_TIME  = True # if you want to see time of load and running model, turn it on!
    
    # instantiate InferenceClass (loading model and creating getScore method)
    t0 = time.time()
    infr_cls = InferenceClass(model_dir = MODEL_PATH)
    model_load_time = time.time() - t0
    

    # predict on batch by getScore method
    t0 = time.time()
    cls_prob = infr_cls.getScore(imgs=IMG_DIR)
    model_run_time =  time.time() - t0
    
    # show result
    cls_prob_pd = pd.DataFrame(cls_prob)
    cls_prob_pd.columns = ['number_prob','alphabet_prob']
    print('calss probabilites are:\n ')
    print(cls_prob_pd)
    
    # show time
    if SHOW_TIME:
        print(f"model loaded in {round(model_load_time,2) } sec")
        print(f"model run on sample in {round(model_run_time*1000,1)} ms")