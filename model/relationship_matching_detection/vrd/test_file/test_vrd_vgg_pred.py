from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import cv2
import json
import random
from model.config import cfg 
from model.ass_fun import *
from net.vtranse_vgg import VTranse
import copy
N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA
N_each_batch = cfg.VRD_BATCH_NUM

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = cfg.DIR + 'input/vrd_roidb.npz'
model_path = cfg.DIR + 'pred_para/vrd_vgg/vrd_vgg0010.ckpt'
save_path = cfg.DIR + 'pred_res/vrd_pred_roidb.npz'
image_name = np.load('/home/cuienjie/vrd/test_file/realtionship_image_name.npy')

relationship_path = '/home/cuienjie/vrd/test_file/predicates.json'

with open(relationship_path,'r') as f:
	relationship_calss = json.load(f)

roidb_read = read_roidb(roidb_path)
test_roidb_ori = roidb_read['test_roidb']

saver = tf.train.Saver()
m = 0

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver.restore(sess, model_path)

	image_path_use = '/home/cuienjie/erd/images/'
	image_name = list(set(image_name))
	test_roidb = []
	for i in range(len(image_name)):
		image_full_path = image_path_use + image_name[i]
		im = cv2.imread(image_full_path)
		if type(im) == type(None):
			continue
		im_shape = np.shape(im)
		im_h = im_shape[0]
		im_w = im_shape[1]

		test_roidb_random = test_roidb_ori[random.randint(0,len(test_roidb_ori)-1)].copy()
		# print(test_roidb_random)
		test_roidb_random['image'] = image_full_path
		test_roidb_random['width'] = im_w
		test_roidb_random['height'] = im_h
		# print(test_roidb_random)
		test_roidb.append(test_roidb_random)
	# if i == 1:
	# 	break
	print(len(test_roidb))

	pred_roidb = []
	relationship_predict = {}

	for roidb_id in range(len(image_name)):
		relationship_tuple = []
		m += 1
		# if m == 1:
		print(m)

		roidb_use = test_roidb[roidb_id]
		# if len(roidb_use['rela_gt']) == 0:
		# 	pred_roidb.append({})
		# 	continue

		pred_rela, pred_rela_score = vnet.test_predicate(sess, roidb_use)
		pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
							'sub_box_dete': roidb_use['sub_box_gt'], 'obj_box_dete': roidb_use['obj_box_gt'],
							'sub_dete': roidb_use['sub_gt'], 'obj_dete': roidb_use['obj_gt']}
		# print(pred_roidb_temp)

		predict_relation = pred_roidb_temp['pred_rela']
		# print(predict_relation)

		predict_relation = predict_relation.astype(int)
		predict_relation = list(set(predict_relation))
		# print(predict_relation)

		key_image_name = str(roidb_use['image']).split('/')[-1]
		print(key_image_name)

		for j in range(len(predict_relation)):
			relationship_tuple.append(str(relationship_calss[predict_relation[j]]))
		print(relationship_tuple)

		relationship_predict[key_image_name] = relationship_tuple
		
print(len(relationship_predict))

path = '/home/cuienjie/vrd/test_file/relationship_predict.json'
with open(path,'w') as f:
	json.dump(relationship_predict,f)
print('save success')
