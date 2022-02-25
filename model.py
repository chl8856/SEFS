import pandas as pd
import numpy as np
import tensorflow as tf

import random
import os

def log(x): 
    return tf.log(x + 1e-8)

def div(x, y):
    return tf.div(x, (y + 1e-8))
    
def xavier_initialization(size):
    dim_ = size[0]
    xavier_stddev = 1. / tf.sqrt(dim_ / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + tf.math.erf(x / tf.math.sqrt(2.))) 

def Gaussian_INVCDF(x): #change from erf function
    return tf.math.sqrt(2.) * erfinv(2.*x - 1)

def erfinv(x):
    import math
    PI = tf.constant(math.pi, dtype=tf.float32)    
    return tf.math.sqrt(PI)/2. * (0.2617993877991494*x**3 + 0.14393173084921979*x**5 + 0.09766361950392055*x**7 + 0.07329907936638086*x**9)


### DEFINE PREDICTOR
def predictor(x_, o_dim_, o_type_, num_layers_=1, h_dim_=100, activation_fn=tf.nn.relu, keep_prob_=1.0, w_reg_=None, reuse=tf.AUTO_REUSE):
    '''
        x_            : (2D-tensor) input
        o_dim_        : (int) output dimension
        o_type_       : (string) output type one of {'continuous', 'categorical', 'binary'}
        num_layers_   : (int) # of hidden layers
        activation_fn_: tf activation functions
        reuse         : (bool) 
    '''
    if o_type_ == 'continuous':
        out_fn = None
    elif o_type_ == 'categorical':
        out_fn = tf.nn.softmax #for classification task
    elif o_type_ == 'binary':
        out_fn = tf.nn.sigmoid
    else:
        raise ValueError('Wrong output type. The value {}!!'.format(o_type_))

    with tf.variable_scope('predictor', reuse=reuse):
        if num_layers_ == 1:
            out =  tf.contrib.layers.fully_connected(inputs=x_, num_outputs=o_dim_, activation_fn=out_fn, weights_regularizer=w_reg_, scope='predictor_out')
        else: #num_layers > 1
            for tmp_layer in range(num_layers_-1):
                if tmp_layer == 0:
                    net = x_
                net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, weights_regularizer=w_reg_, scope='predictor_'+str(tmp_layer))
                net = tf.nn.dropout(net, keep_prob=keep_prob_)
            out =  tf.contrib.layers.fully_connected(inputs=net, num_outputs=o_dim_, activation_fn=out_fn, weights_regularizer=w_reg_, scope='predictor_out')  
    return out


### DEFINE SUPERVISED LOSS FUNCTION
def loss_y(y_true_, y_pred_, y_type_):                
    if y_type_ == 'continuous':
        tmp_loss = tf.reduce_sum((y_true_ - y_pred_)**2, axis=-1)
    elif y_type_ == 'categorical':
        tmp_loss = - tf.reduce_sum(y_true_ * log(y_pred_), axis=-1)
    elif y_type_ == 'binary':
        tmp_loss = - tf.reduce_sum(y_true_ * log(y_pred_) + (1.-y_true_) * log(1.-y_pred_), axis=-1)
    else:
        raise ValueError('Wrong output type. The value {}!!'.format(y_type_))                    
    return tmp_loss


def fcnet(x_, o_dim_, o_fn_, num_layers_=1, h_dim_=100, activation_fn=tf.nn.relu, keep_prob_=1.0, w_reg_=None, name='fcnet', reuse=tf.AUTO_REUSE):
    '''
        x_            : (2D-tensor) input
        o_dim_        : (int) output dimension
        o_type_       : (string) output type one of {'continuous', 'categorical', 'binary'}
        num_layers_   : (int) # of hidden layers
        activation_fn_: tf activation functions
        reuse         : (bool) 
    '''
    with tf.variable_scope(name, reuse=reuse):
        if num_layers_ == 1:
            out =  tf.contrib.layers.fully_connected(inputs=x_, num_outputs=o_dim_, activation_fn=o_fn_, weights_regularizer=w_reg_, scope='layer_out')
        else: #num_layers > 1
            for tmp_layer in range(num_layers_-1):
                if tmp_layer == 0:
                    net = x_
                net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, weights_regularizer=w_reg_, scope='layer_'+str(tmp_layer))
                net = tf.nn.dropout(net, keep_prob=keep_prob_)
            out =  tf.contrib.layers.fully_connected(inputs=net, num_outputs=o_dim_, activation_fn=o_fn_, weights_regularizer=w_reg_, scope='layer_out')  
    return out



class SEFS_SS_Phase: 
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess             = sess
        self.name             = name
        
        # INPUT/OUTPUT DIMENSIONS
        self.x_dim            = input_dims['x_dim']
        self.z_dim            = input_dims['z_dim']  # dimension of the encoding        
        
        self.reg_scale        = network_settings['reg_scale'] #regularization for the network       
        
        self.h_dim_e          = network_settings['h_dim_e']
        self.num_layers_e     = network_settings['num_layers_e'] #predictor layers        
        
        self.h_dim_d          = network_settings['h_dim_d']
        self.num_layers_d     = network_settings['num_layers_d'] #decoder layers        
        
        self.fc_activate_fn   = network_settings['fc_activate_fn']
        self.reg_scale        = network_settings['reg_scale']
        
        self._build_net()

        
    def _build_net(self):
        with tf.variable_scope(self.name):            
            self.lr_rate        = tf.placeholder(tf.float32, name='learning_rate')   #predictor      
            self.k_prob         = tf.placeholder(tf.float32, name='keep_probability') 
            
            self.alpha          = tf.placeholder(tf.float32, name='ss_phase_coeff')
            
            ### INPUT/OUTPUT
            self.x              = tf.placeholder(tf.float32, [None, self.x_dim], name='input')
            self.x_bar          = tf.placeholder(tf.float32, [None, self.x_dim], name='input_perturbed')
            
            self.mb_size        = tf.shape(self.x)[0]
            
            self.m              = tf.placeholder(tf.float32, [None, self.x_dim], name='mask')
            
            self.x_tilde        = tf.multiply(self.x, self.m) + tf.multiply(self.x_bar, 1. - self.m)
            self.m_tilde        = tf.cast(tf.equal(self.x, self.x_tilde), dtype=tf.float32) #this will be used as the true label!
            
       
            self.z              = fcnet(
                x_=self.x_tilde,  o_dim_=self.z_dim, o_fn_=tf.nn.relu, 
                num_layers_=self.num_layers_e, h_dim_=self.h_dim_e, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn, name='encoder', 
            )          
            
            self.x_hat          = fcnet(
                x_=self.z,  o_dim_=self.x_dim, o_fn_=tf.nn.sigmoid, 
                num_layers_=self.num_layers_d, h_dim_=self.h_dim_d, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn, name='decoder_x'
            )          
            self.m_hat          = fcnet(
                x_=self.z,  o_dim_=self.x_dim, o_fn_=tf.nn.sigmoid, 
                num_layers_=self.num_layers_d, h_dim_=self.h_dim_d, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn, name='decoder_m'
            )
            
            
            ### DEFINE VARIABLES/LOSSES/OPTIMIZERS
            #VARIABLES
            self.vars_encoder  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/encoder')
            self.vars_decoder  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/decoder_x') +\
                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/decoder_m')
            
                                    
            vars_reg_g          = [w for w in self.vars_encoder + self.vars_decoder if 'weights' in w.name]
            regularizer_g       = tf.contrib.layers.l1_regularizer(scale=self.reg_scale, scope=None)
                        
            #LOSS
            loss_reg_g          = tf.contrib.layers.apply_regularization(regularizer_g, vars_reg_g)          
                        
            self.loss_RECON_x   = tf.reduce_mean(tf.reduce_sum((self.x - self.x_hat)**2, axis=1))
            self.loss_RECON_m   = tf.reduce_mean(- tf.reduce_sum(self.m_tilde * log(self.m_hat) + (1.-self.m_tilde) * log(1.-self.m_hat), axis=1))            
            self.loss_MAIN      = self.loss_RECON_x + self.alpha * self.loss_RECON_m + loss_reg_g

            
            grad_encoder        = tf.gradients(ys=self.loss_MAIN, xs=self.vars_encoder)
            grad_decoder        = tf.gradients(ys=self.loss_MAIN, xs=self.vars_decoder)
                        
            opt_MAIN             = tf.train.AdamOptimizer(learning_rate =self.lr_rate)
            self.solver_MAIN     = opt_MAIN.apply_gradients(
                grads_and_vars=[(grad, var) for grad,var in zip(grad_encoder+grad_decoder, self.vars_encoder+self.vars_decoder)]
            )
            
    
    def train_main(self, x_, x_bar_, m_, alpha_, lr_train_=1e-3, k_prob_=1.0):
        return self.sess.run([self.solver_MAIN, self.loss_MAIN, self.loss_RECON_x, self.loss_RECON_m],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.m:m_,
                                        self.alpha:alpha_, 
                                        self.lr_rate: lr_train_, 
                                        self.k_prob: k_prob_})
    
    def get_loss_main(self, x_, x_bar_, m_, alpha_):
        return self.sess.run([self.loss_MAIN, self.loss_RECON_x, self.loss_RECON_m],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.m:m_, 
                                        self.alpha:alpha_,
                                        self.k_prob: 1.0})
    
    
    def predict_x_and_m(self, x_, x_bar_, m_):        
        return self.sess.run([self.x_hat, self.m_hat],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, 
                                        self.m:m_, 
                                        self.k_prob:1.0})
    
    def get_z(self, x_, x_bar_, m_):        
        return self.sess.run(self.z,
                             feed_dict={self.x:x_, self.x_bar:x_bar_,
                                        self.m:m_, 
                                        self.k_prob:1.0})
    
    

    

#### DEFINE PROPOSED-NETWORK
class SEFS_S_Phase:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess             = sess
        self.name             = name
                
        # INPUT/OUTPUT DIMENSIONS
        self.x_dim            = input_dims['x_dim']
        self.z_dim            = input_dims['z_dim']  # dimension of the encoding        
        self.y_dim            = input_dims['y_dim']
        self.y_type           = input_dims['y_type']
        
        if self.y_type == 'continuous':
            self.out_fn = None
        elif self.y_type == 'categorical':
            self.out_fn = tf.nn.softmax #for classification task
        elif self.y_type == 'binary':
            self.out_fn = tf.nn.sigmoid
        else:
            raise ValueError('Wrong output type. The value {}!!'.format(self.y_type))

        
        self.reg_scale        = network_settings['reg_scale'] #regularization for the network       
        
        self.h_dim_e          = network_settings['h_dim_e']
        self.num_layers_e     = network_settings['num_layers_e'] #encoder layers        
        self.fc_activate_fn_e = network_settings['fc_activate_fn_e'] #encoder activation 
        
        self.h_dim_p          = network_settings['h_dim_p']
        self.num_layers_p     = network_settings['num_layers_p'] #predictor layers                
        self.fc_activate_fn_p = network_settings['fc_activate_fn_p'] #predictor activation
        
        self.reg_scale        = network_settings['reg_scale']
        
        self._build_net()

        
    def _build_net(self):
        with tf.variable_scope(self.name):            
            self.lr_rate        = tf.placeholder(tf.float32, name='learning_rate') 
                        
            self.k_prob         = tf.placeholder(tf.float32, name='keep_probability')
            
            self.lmbda          = tf.placeholder(tf.float32, name='coef_l0penalty')
            
            ### INPUT/OUTPUT
            self.x              = tf.placeholder(tf.float32, [None, self.x_dim], name='input')
            self.x_bar          = tf.placeholder(tf.float32, [None, self.x_dim], name='input_perturbed')
            self.y              = tf.placeholder(tf.float32, [None, self.y_dim], name='output')
            
            self.q              = tf.placeholder(tf.float32, [None, self.x_dim], name='copula_noise')
            
            self.mb_size        = tf.shape(self.x)[0]
                
            self.mask_final     = tf.placeholder(tf.float32, [self.x_dim], name='mask_final')
                               
            
            ### RELAXATION -- GATE VECTOR GENERATION
            self.pi_logit       = tf.Variable(tf.zeros([self.x_dim]), dtype=tf.float32, name='feature_probability')
            self.pi             = tf.sigmoid(self.pi_logit)
            
            ### generate multivariate Bernoulli variable
            self.u              = Gaussian_CDF(self.q)    
            self.m              = tf.nn.sigmoid(log(self.pi) - log(1.-self.pi) + log(self.u) - log(1.-self.u))
                    
            self.x_tilde        = tf.multiply(self.x, self.m) #+ tf.multiply(self.x_bar, 1. - self.m)           
            self.x_final        = tf.multiply(self.x, self.mask_final) #+ tf.multiply(self.x_bar, 1. - self.mask_final)
            
            self.z              = fcnet(
                x_=self.x_tilde,  o_dim_=self.z_dim, o_fn_=tf.nn.relu, 
                num_layers_=self.num_layers_e, h_dim_=self.h_dim_e, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn_e, name='encoder', 
            )
            self.z_final        = fcnet(
                x_=self.x_final,  o_dim_=self.z_dim, o_fn_=tf.nn.relu, 
                num_layers_=self.num_layers_e, h_dim_=self.h_dim_e, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn_e, name='encoder', reuse=True
            )

            self.y_pred              = fcnet(
                x_=self.z,  o_dim_=self.y_dim, o_fn_=self.out_fn, 
                num_layers_=self.num_layers_p, h_dim_=self.h_dim_p, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn_p, name='predictor', 
            )
            self.y_final             = fcnet(
                x_=self.z_final,  o_dim_=self.y_dim, o_fn_=self.out_fn, 
                num_layers_=self.num_layers_p, h_dim_=self.h_dim_p, 
                keep_prob_=self.k_prob, activation_fn=self.fc_activate_fn_p, name='predictor', reuse=True
            )
             

            ### DEFINE VARIABLES/LOSSES/OPTIMIZERS
            #VARIABLES
            self.vars_selector     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/feature_probability')
                        
            self.vars_encoder      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/encoder')
            self.vars_encoder_last = self.vars_encoder[-2:]
            
            self.vars_predictor    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/predictor')
            
            vars_reg_p             = [w for w in self.vars_predictor if 'weights' in w.name]
            regularizer_p          = tf.contrib.layers.l1_regularizer(scale=self.reg_scale, scope=None)
                        
            #LOSS
            loss_reg_p             = tf.contrib.layers.apply_regularization(regularizer_p, vars_reg_p)       
            loss_main              = loss_y(y_pred_=self.y_pred, y_true_=self.y, y_type_=self.y_type)
            
            self.loss_m0           = tf.reduce_mean(self.pi) #a basic version w/o considering the interactions.
            self.LOSS              = tf.reduce_mean(loss_main) + self.lmbda * self.loss_m0 + loss_reg_p
                                    
            
            self.vars_all          = self.vars_encoder+self.vars_predictor+self.vars_selector
            self.vars_finetune     = self.vars_encoder_last+self.vars_predictor+self.vars_selector
            self.vars_noencoder    = self.vars_predictor+self.vars_selector
            
            grad_main_all           = tf.gradients(ys=self.LOSS, xs=self.vars_all)
            grad_main_finetune      = tf.gradients(ys=self.LOSS, xs=self.vars_finetune)
            grad_main_noencoder     = tf.gradients(ys=self.LOSS, xs=self.vars_noencoder)
            
            opt_main_noencoder      = tf.train.AdamOptimizer(learning_rate =self.lr_rate)            
            self.solver_noencoder   = opt_main_noencoder.apply_gradients(
                grads_and_vars=[(grad, var) for grad,var in zip(grad_main_noencoder, self.vars_noencoder)]
            )
            opt_main_all            = tf.train.AdamOptimizer(learning_rate =self.lr_rate)
            self.solver_all         = opt_main_all.apply_gradients(
                grads_and_vars=[(grad, var) for grad,var in zip(grad_main_all, self.vars_all)]
            )
            opt_main_finetune       = tf.train.AdamOptimizer(learning_rate =self.lr_rate)
            self.solver_finetune    = opt_main_finetune.apply_gradients(
                grads_and_vars=[(grad, var) for grad,var in zip(grad_main_finetune, self.vars_finetune)]
            )            

                
    def train_noencoder(self, x_, x_bar_, y_, q_, lmbda_=1e-3, lr_train_=1e-3, k_prob_=1.0):
        return self.sess.run([self.solver_noencoder, self.LOSS, self.loss_m0],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.y:y_, self.q:q_,
                                        self.lmbda: lmbda_,
                                        self.lr_rate: lr_train_, 
                                        self.k_prob: k_prob_})
    
    def train_finetune(self, x_, x_bar_, y_, q_, lmbda_=1e-3, lr_train_=1e-3, k_prob_=1.0):
        return self.sess.run([self.solver_finetune, self.LOSS, self.loss_m0],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.y:y_, self.q:q_,
                                        self.lmbda: lmbda_,
                                        self.lr_rate: lr_train_, 
                                        self.k_prob: k_prob_})
    
    def train_all(self, x_, x_bar_, y_, q_, lmbda_=1e-3, lr_train_=1e-3, k_prob_=1.0):
        return self.sess.run([self.solver_all, self.LOSS, self.loss_m0],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.y:y_, self.q:q_,
                                        self.lmbda: lmbda_,
                                        self.lr_rate: lr_train_, 
                                        self.k_prob: k_prob_})
    
    def get_loss(self, x_, x_bar_, y_, q_, lmbda_=1e-3):
        return self.sess.run([self.LOSS, self.loss_m0],
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.y:y_, self.q:q_,
                                        self.lmbda: lmbda_,
                                        self.k_prob: 1.0})
    
    
    def get_z(self, x_, x_bar_, q_):        
        return self.sess.run(self.z,
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.q:q_, 
                                        self.k_prob:1.0})
    
    def get_z_final(self, x_, x_bar_, m_):        
        return self.sess.run(self.z_final,
                             feed_dict={self.x:x_, self.x_bar:x_bar_,
                                        self.mask_final:m_,
                                        self.k_prob:1.0})
    

    def predict(self, x_, x_bar_, q_):        
        return self.sess.run(self.y_pred,
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.q:q_,
                                        self.k_prob:1.0})
    
    def predict_final(self, x_, x_bar_, m_):        
        return self.sess.run(self.y_final,
                             feed_dict={self.x:x_, self.x_bar:x_bar_, self.mask_final:m_, 
                                        self.k_prob:1.0})