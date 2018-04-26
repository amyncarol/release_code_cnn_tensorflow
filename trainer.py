import tensorflow as tf
import datetime
import os
import sys
import argparse

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):

     
        self.net = net
        self.data = data
       
        #Number of iterations to train for
        self.max_iter = 5000
        #Every 200 iterations please record the test and train loss
        self.summary_iter = 200
        


        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)
        

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    


    def optimize(self):

        #Append the current train and test accuracy every 200 iterations
        self.train_accuracy = []
        self.test_accuracy = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        '''
        for step in range(self.max_iter):
            images_batch, labels_batch = self.data.get_train_batch()
            _, loss = self.sess.run([self.train, self.net.class_loss], feed_dict={self.net.images:images_batch, self.net.labels:labels_batch})

            print('Step {}: current loss is {}'.format(step, loss))

            if step % self.summary_iter == 0:
                images_batch, labels_batch = self.data.get_train_batch()
                train_accuracy = self.sess.run([self.net.accuracy], feed_dict={self.net.images:images_batch, self.net.labels:labels_batch})
                self.train_accuracy.append(train_accuracy)

                images_batch, labels_batch = self.data.get_validation_batch()
                test_accuracy = self.sess.run([self.net.accuracy], feed_dict={self.net.images:images_batch, self.net.labels:labels_batch})
                self.test_accuracy.append(test_accuracy)

   
