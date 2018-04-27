import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import confusion_matrix
import random
import cv2
import IPython
import numpy as np
import tensorflow as tf


class Viz_Feat(object):


    def __init__(self,val_data,train_data, class_labels,sess=None):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess





    def vizualize_features(self,net):

        images = [0,10,100]
        '''
        Compute the response map for the index images
        '''
        for i in images:
            ##feeding image
            image = self.val_data[i]['features']
            image_batch = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]))
            image_batch[0] = image
            class_idx = np.argmax(self.val_data[i]['label'])
            # print(self.val_data[i]['label'])
            # print(class_idx)
            # print(self.CLASS_LABELS[class_idx])

            ##plot the original image
            plt.imshow(self.val_data[i]['c_img'])
            plt.title(self.CLASS_LABELS[class_idx])
            plt.savefig(str(i)+'.png')

            ##run responce map
            response_map_t = tf.get_default_graph().get_tensor_by_name("yolo/conv:0")
            response_map = self.sess.run(response_map_t, feed_dict={net.images:image_batch})

            ##plot responce map
            for j in range(5):
                plt.imshow(self.revert_image(response_map[0, :, :, j]))
                plt.title(self.CLASS_LABELS[class_idx])
                plt.savefig(str(i)+'_'+str(j)+'.png')
      

    def revert_image(self,img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img

        




