import sys
import numpy as np
import tensorflow as tf


class adversarial_sample_generator(tf.Module):
    def __init__(self,model,loss_fn,input_shape,norm,clip_min=tf.constant(0.),clip_max=tf.constant(255.)):
        super(adversarial_sample_generator,self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.reduction_axes = tf.constant([1,2,3])
        self.avoid_zero = tf.constant(1e-12)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.expected_batch_shape = input_shape
        self.expected_batch_size = input_shape[0]
        self.expected_input_shape = input_shape[1:]
        if norm == np.inf:
            self.norm = tf.constant(-1.)
        else:
            self.norm = tf.constant(2.)
    
    def optimize_linear(self,gradients,epsilon):
        if self.norm == tf.constant(-1.):
            optimal_perturbations = tf.sign(gradients)
            scaled_perturbations = tf.multiply(epsilon,optimal_perturbations)
            return scaled_perturbations
        else:
            squared_gradients = tf.square(gradients)
            sum_of_squares = tf.reduce_sum(squared_gradients,self.reduction_axes,keepdims=True)
            non_zero_sum_of_squares = tf.maximum(self.avoid_zero,sum_of_squares)
            optimal_perturbations = gradients / tf.sqrt(non_zero_sum_of_squares)
            scaled_perturbations = tf.multiply(epsilon,optimal_perturbations)
            return scaled_perturbations

    def random_lp_vector(self,epsilon):
        if self.norm == tf.constant(-1.):
            random_vector = tf.random.uniform(self.expected_batch_shape,-epsilon,epsilon,dtype=tf.float32)
            return random_vector
        else:
            random_vector = tf.random.uniform(self.expected_batch_shape,-tf.sqrt(epsilon),tf.sqrt(epsilon),dtype=tf.float32)
            norm_bound_random_vector = self.optimize_linear(random_vector,epsilon)
            random_scaling = tf.random.uniform([1],0.,1.,dtype=tf.float32)
            scaled_random_vector = tf.multiply(random_scaling,norm_bound_random_vector)
            return scaled_random_vector
            
    def clip_eta(self,eta,epsilon):
        if self.norm == tf.constant(-1.):
            eta = tf.clip_by_value(eta,-epsilon,epsilon)
            return eta
        else:
            norm = tf.sqrt(tf.maximum(self.avoid_zero,tf.reduce_sum(tf.square(eta),self.reduction_axes,keepdims=True)))
            scaling_factor = tf.minimum(1.0,tf.math.divide(epsilon,norm))
            eta = eta * scaling_factor
            return eta
            
    def __call__(self,x):
        y_pred = self.model(x)
        return y_pred
    
    def calculate_gradients(self,x,y):
        with tf.GradientTape() as g:
            g.watch(x)
            y_pred = self(x)
            loss = self.loss_fn(y,y_pred)
        gradients = g.gradient(loss,x)
        return gradients
        
    def set_FGM_parameters(self,epsilon):
        self.FGM_epsilon = epsilon
    
    def set_PGD_parameters(self,epsilon,step_size,num_steps,rand_init):
        self.PGD_epsilon = epsilon
        self.PGD_step_size = step_size
        self.PGD_num_steps = num_steps
        self.PGD_rand_init = rand_init
    
    @tf.function
    def generate_samples_FGM(self,x,y):
        gradients = self.calculate_gradients(x,y)
        optimal_perturbation = self.optimize_linear(gradients,self.FGM_epsilon)
        adv_x = x + optimal_perturbation
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        return adv_x
    
    def perform_PGD_step(self,x,y):
        gradients = self.calculate_gradients(x,y)
        optimal_perturbation = self.optimize_linear(gradients,self.PGD_step_size)
        adv_x = x + optimal_perturbation
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        return adv_x

    @tf.function
    def generate_samples_PGD(self,x,y):
        if self.PGD_rand_init:
            eta = self.random_lp_vector(self.PGD_epsilon)
        else:
            eta = tf.zeros_like(x)
        adv_x = x + eta
        adv_x = tf.clip_by_value(adv_x,self.clip_min,self.clip_max)
        for _ in tf.range(self.PGD_num_steps):
            adv_x = self.perform_PGD_step(adv_x,y)
            eta = adv_x - x
            eta = self.clip_eta(eta,self.PGD_epsilon)
            adv_x = x + eta
            adv_x = tf.clip_by_value(adv_x,self.clip_min,self.clip_max)
        return adv_x

