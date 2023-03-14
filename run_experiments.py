import sys
import os
import gc
import csv
import glob
import argparse
import pathlib

import numpy as np
import tensorflow as tf
from attack_model import adversarial_sample_generator
sys.path.append("../")
from timeit import default_timer as timer

def load_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img,channels=3)
    img = tf.cast(img,tf.float32)
    img = tf.image.resize(img, [224, 224])
    return img

def create_one_hot_label(class_num):
    return tf.one_hot(class_num,1000)

def process_path(file_path):
    img = load_img(file_path)
    label = create_one_hot_label(int(tf.strings.split(file_path, os.sep)[-2]))
    return img, label

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('-data_path', default='../data', type=str)
parser.add_argument('-batch_size', default=1, type=int)
parser.add_argument('-model', default='resnet', type=str, choices=["resnet","densenet","inception","xception"])
parser.add_argument('-input_size', default=224, type=int, choices=[224])
parser.add_argument('-norm', default='inf', type=str, choices=['inf','2'])
parser.add_argument('-base_output_path', default='../results', type=str)
parser.add_argument('-run_num', default=0, type=int)
parser.add_argument('-cpu_or_gpu', default='cpu', type=str, choices=['cpu','gpu'])
parser.add_argument('-mixed_precision',default=0,type=int,choices=[0,1])
parser.add_argument('-batch_corrected', action='store_true')
args = parser.parse_args()

mixed_precision = args.mixed_precision
if mixed_precision:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    precision_flag = "true"
else:
    precision_flag = "false"
data_path = pathlib.Path(args.data_path)
batch_size = args.batch_size
model = args.model
input_layer = tf.keras.layers.Input([args.input_size,args.input_size,3],batch_size=batch_size,dtype=tf.float32)
if args.batch_corrected:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
if args.norm == "inf":
    norm = np.inf
    norm_name = "L-INF"
    epsilon = tf.constant(8.)
elif args.norm == '2':
    norm = 2
    norm_name = "L-2"
    epsilon = tf.constant(128.)
num_steps = tf.constant(32,dtype=tf.int32)
step_size = (epsilon*2.)/tf.cast(num_steps,tf.float32)
output_path = os.path.join(args.base_output_path,f"Batch_Corrected_{args.batch_corrected}_Mixed_Precision_{precision_flag}_{args.cpu_or_gpu}_{model}_norm_{norm_name}_batchSize_{batch_size}_run_{args.run_num}.csv")
pgd_rand_init = tf.constant(True,dtype=tf.bool)

# Load our model and set up the appropriate preprocessing
if model == "resnet":
    preprocess_layer = tf.keras.applications.resnet.preprocess_input(input_layer)
    model = tf.keras.applications.resnet.ResNet50(input_tensor=preprocess_layer)
elif model == "densenet":
    preprocess_layer = tf.keras.applications.densenet.preprocess_input(input_layer)
    model = tf.keras.applications.densenet.DenseNet121(input_tensor=preprocess_layer)
elif model == "inception":
    preprocess_layer = tf.keras.applications.inception_v3.preprocess_input(input_layer)
    model = tf.keras.applications.inception_v3.InceptionV3(input_tensor=preprocess_layer)
elif model == "xception":
    preprocess_layer = tf.keras.applications.xception.preprocess_input(input_layer)
    model = tf.keras.applications.xception.Xception(input_tensor=preprocess_layer)
else:
    print("Select a valid architecture")
    sys.exit()
model.trainable=False

# Set up our adversarial sample generator
adv_generator = adversarial_sample_generator(model,loss_fn,input_layer.shape,norm)
adv_generator.set_FGM_parameters(epsilon)
adv_generator.set_PGD_parameters(epsilon,step_size,num_steps,pgd_rand_init)

# Set up our data loaders
list_ds = tf.data.Dataset.list_files(str(data_path/'*/*'))
labeled_ds = list_ds.map(process_path)
shuffled_ds = labeled_ds.shuffle(10000,seed=42,reshuffle_each_iteration=False)
batched_ds = shuffled_ds.batch(batch_size)

# Loop through our dataset in batches
print("\nBeginning Results Generation")
results = []
results.append(["loading_time","prediction_time","original_num_incorrect","FGSM_time","FGSM_num_successes","PGD_time","PGD_num_successes","total_time"])
for batch in batched_ds:
    # Record how long it takes to load our batch of images and labels
    start = timer()
    imgs = batch[0]
    labels = batch[1]
    end = timer()
    loading_time = end - start
    if len(imgs) != batch_size:
        break
    # See how long it takes us to perform inferrence on the current batch and record how many we got wrong
    start = timer()
    original_predictions = model(imgs)
    end = timer()
    prediction_time = end - start
    original_num_incorrect = tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.math.argmax(original_predictions,axis=1),tf.math.argmax(labels,axis=1)),tf.int32)).numpy()
    
    # Perform FGSM and record how long it took as well as how successful it was (at making the model fail)
    start = timer()
    adv_imgs = adv_generator.generate_samples_FGM(x=imgs,y=labels)
    end = timer()
    adv_predictions = model(adv_imgs)
    FGSM_time = end - start
    FGSM_num_successes = tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.math.argmax(adv_predictions,axis=1),tf.math.argmax(labels,axis=1)),tf.int32)).numpy()
    
    # Perform PGD and record how long it took as well as how successful it was (at making the model fail)
    start = timer()
    adv_imgs = adv_generator.generate_samples_PGD(x=imgs,y=labels)
    adv_predictions = model(adv_imgs)
    end = timer()
    PGD_time = end - start
    PGD_num_successes = tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.math.argmax(adv_predictions,axis=1),tf.math.argmax(labels,axis=1)),tf.int32)).numpy()
    
    # Calculate how long this batch took and record the current results
    total_time = loading_time + prediction_time + FGSM_time + PGD_time

    # Update all the results to be divided by batch_size

    batch_result = [loading_time,prediction_time,original_num_incorrect,FGSM_time,FGSM_num_successes,PGD_time,PGD_num_successes,total_time]
    results.append(batch_result)
print("Done with Results Generation\n")
with open(output_path,'w',newline='') as fout:
    writer = csv.writer(fout)
    writer.writerows(results)

