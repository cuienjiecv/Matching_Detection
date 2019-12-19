import json 
import os
import shutil
import numpy as np 


path = "./predict_positive_test.json"
with open(path,'r') as f:
	predict_positive = json.load(f)
print(len(predict_positive))
# print(predict_positive[0])
image_path = 'E:/程序总结/matching detection/data/RefCOCO+/images/'
relationship_images_path = 'E:/程序总结/matching detection/model/relationship_matching_detection/relationship_images/'

#{'image_id': '000000269557 1 person in blue vest', 
#'objects': ['person', 'vest'], 'relations': ['person', 'in', 'vest']}

images_name = []
for i in range(len(predict_positive)):
	if len(predict_positive[i]['relations']) != 0:
		# print(predict_positive[i]['image_id'])
		if predict_positive[i]['image_id'].split(' ')[1] == '2':
			image_name = 'COCO_train2014_' + predict_positive[i]['image_id'].split(' ')[0] + '.jpg'
			images_name.append(image_name)

# print(len(images_name))

images_name = list(set(images_name))

print(len(images_name))

# n = 0
# for i in range(len(images_name)):
# 	source_image = image_path + images_name[i]
# 	target_image = relationship_images_path + images_name[i]
# 	shutil.copy(source_image,target_image)
	# n += 1
	# if n == 1:
	# 	break

# images_name_save = './realtionship_image_name.npy'
# np.save(images_name_save, images_name)