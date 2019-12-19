import json
from nltk.corpus import wordnet as wn

NPRefCOCO_path = "E:/程序总结/matching detection/data/parsing_data/NPRefCOCO_parsing.json"
with open(NPRefCOCO_path,'r') as f:
	NPRefCOCO = json.load(f)
print(len(NPRefCOCO))

# 选择不同的长度
NPRefCOCO_length = []
for i in range(len(NPRefCOCO)):
	if len(NPRefCOCO[i]['image_id'].split(' ')[2:]) >= 4 :
		# print(NPRefCOCO[i])
		NPRefCOCO_length.append(NPRefCOCO[i])

# 先整理数据
# 1、提取objects 和 relation
#    提取image_id
# 2、[{'image_id':'image_name label question',
#       'objects':[o1,O2,...],
#       'relation':[o1,r12,O2,O3,r34,O4]},{},{}]
# 3、对于不存在元素的tuple，保存id

# len(ECRD_expression) = 135763
# {'test_tuples': [{'tuple': ['girl'], 'truth_value': False}], 'ref_tuples': [], 
# 'scores': {'All': {'pr': 0, 're': 0, 'f': 0, 'fn': 0, 'numImages': 1, 'fp': 1, 'tp': 0}}, 
# 'image_id': '000000291889 1 girl'}

# clear_items = ['1','2','3','4','5','6','7','8','9','10','color','face','type','day','one','playing','picture','sleeping',
#                 'face/side','photo/picture','photo','walking','these','two','left','time','x','game','fun',
#                 'second','lit','smiling','holding','kind','right','that','year','eating','day','kawasaki','it','name',
#                'can','thing','number','end','running','touching','shadow','holiday','air','what','morning','front',
#                'stand', 'standing', 'india', 'sort', 'this', 'sleeping','barking','weight','red','flying','top','xxxx']

NPCOCO = []
P_empty_index = []
N_empty_index = []

n = 0
for i in range(0,len(NPRefCOCO_length)):
	element = {}
	element_objects = []
	state1 = False
	for j in range(len(NPRefCOCO_length[i]["test_tuples"])):
		if len(NPRefCOCO_length[i]["test_tuples"][j]['tuple']) == 1:
			element_objects = element_objects + NPRefCOCO_length[i]["test_tuples"][j]['tuple']
			state1 = True
				# print(s[i]["test_tuples"][j]['tuple'])
	if state1 == True:
		element['image_id'] = NPRefCOCO_length[i]['image_id']
		element['objects'] = element_objects
	else:
		if NPRefCOCO_length[i]['image_id'].split(' ')[1] == '1':
			P_empty_index.append(NPRefCOCO_length[i]['image_id'])

		if NPRefCOCO_length[i]['image_id'].split(' ')[1] == '2':
			N_empty_index.append(NPRefCOCO_length[i]['image_id'])

	if len(element_objects) >= 2:
		n += 1
		element_relations = []
		for k in range(0,len(NPRefCOCO_length[i]["test_tuples"])):
			if len(NPRefCOCO_length[i]["test_tuples"][k]['tuple']) == 3:
				element_relations = element_relations + NPRefCOCO_length[i]["test_tuples"][k]['tuple']
				break
		element['relations'] = element_relations
	else:
		element['relations'] = []

	if state1 == True:
		NPCOCO.append(element)

print(len(N_empty_index))
# path1 = 'E:/程序总结/problem/P_empty_index.json'
# with open(path1,'w') as f:
# 	json.dump(P_empty_index,f)

# path2 = 'E:/程序总结/problem/N_empty_index.json'
# with open(path2,'w') as f:
# 	json.dump(N_empty_index,f)

# print(empty_index)
print(len(NPCOCO))

# save_path = 'E:/Ubuntu_Share/Question_Relevance_Detection/ECRD/model/NPCOCO.json'
# with open(save_path,'w') as f:
# 	json.dump(NPCOCO,f)

entity_dictionary_path = 'E:/程序总结/matching detection/data/entity_dictionary/NPCOCO_entity_dictionary.json'
with open(entity_dictionary_path,'r') as f:
	entity_dictionary = json.load(f)

# print(entity_dictionary['COCO_train2014_000000457437.jpg'])
count = 0
# 清理ECRD_entity_dictionary中的数据生成entity_dictionary

for key,value in entity_dictionary.items():
	# entity_dictionary[key] = list(set(value))
	# print(key,value)
	count += 1
	if 'dining table' in value[0]:
		entity_dictionary[key][0].remove('dining table')
		entity_dictionary[key][0].append('dining')
		entity_dictionary[key][0].append('table')
	if 'traffic light' in value[0]:
		entity_dictionary[key][0].remove('traffic light')
		entity_dictionary[key][0].append('traffic')
		entity_dictionary[key][0].append('light')
	if 'fire hydrant' in value[0]:
		entity_dictionary[key][0].remove('fire hydrant')
		entity_dictionary[key][0].append('fire')
		entity_dictionary[key][0].append('hydrant')
	if 'stop sign' in value[0]:
		entity_dictionary[key][0].remove('stop sign')
		entity_dictionary[key][0].append('stop')
		entity_dictionary[key][0].append('sign')
	if 'parking meter' in value[0]:
		entity_dictionary[key][0].remove('parking meter')
		entity_dictionary[key][0].append('parking')
		entity_dictionary[key][0].append('meter')
	if 'sports ball' in value[0]:
		entity_dictionary[key][0].remove('sports ball')
		entity_dictionary[key][0].append('sports')
		entity_dictionary[key][0].append('ball')
	if 'baseball bat' in value[0]:
		entity_dictionary[key][0].remove('baseball bat')
		entity_dictionary[key][0].append('baseball')
		entity_dictionary[key][0].append('bat')
	if 'baseball glove' in value[0]:
		entity_dictionary[key][0].remove('baseball glove')
		entity_dictionary[key][0].append('baseball')
		entity_dictionary[key][0].append('glove')
	if 'tennis racket' in value[0]:
		entity_dictionary[key][0].remove('tennis racket')
		entity_dictionary[key][0].append('tennis')
		entity_dictionary[key][0].append('racket')	
	if 'wine glass' in value:
		entity_dictionary[key][0].remove('wine glass')
		entity_dictionary[key][0].append('wine')
		entity_dictionary[key][0].append('glass')
	if 'hot dog' in value[0]:
		entity_dictionary[key][0].remove('hot dog')
		entity_dictionary[key][0].append('hot')
		entity_dictionary[key][0].append('dog')
	if 'potted plant' in value[0]:
		entity_dictionary[key][0].remove('potted plant')
		entity_dictionary[key][0].append('potted')
		entity_dictionary[key][0].append('plant')	
	if 'dining table' in value[0]:
		entity_dictionary[key][0].remove('dining table')
		entity_dictionary[key][0].append('dining')
		entity_dictionary[key][0].append('table')
	if 'cell phone' in value[0]:
		entity_dictionary[key][0].remove('cell phone')
		entity_dictionary[key][0].append('cell')
		entity_dictionary[key][0].append('phone')		
	if 'teddy bear' in value[0]:
		entity_dictionary[key][0].remove('teddy bear')
		entity_dictionary[key][0].append('teddy')
		entity_dictionary[key][0].append('bear')	
	if 'hair drier' in value[0]:
		entity_dictionary[key][0].remove('hair drier')
		entity_dictionary[key][0].append('hair')
		entity_dictionary[key][0].append('drier')		

# print(entity_dictionary['COCO_train2014_000000457437.jpg'])
# len(ECRD_expression) = 135763
# {"image_id": "000000420400 2 man with do rag and red frisbee",
#  "objects": ["frisbee", "man", "rag"], "relations": ["man", "do", "frisbee", "man", "do", "rag"]

# # len(ECRD_entity_dictionary) = 19183
# # 'COCO_train2014_000000191806.jpg': ['microwave', 'oven', 'sink']


# data = NPCOCO[0:int(len(NPCOCO))]
# data = NPCOCO[0:int(len(NPCOCO)*2/3)]

data = NPCOCO[int(len(NPCOCO)*(2/3)):len(NPCOCO)]

def wordnet_syn(word):
	try:
		syn = wn.synsets(word)
	except IndexError:
		syn = []
	return syn

def wordnet_com(syn1,syn2):
	try:
		syn_common = syn1.lowest_common_hypernyms(syn2)
	except IndexError:
		syn_common = []
	return syn_common


right = 0
wrong = 0
TP = 0 #判匹配并且正确的
FP = 0 #判匹配并且错误的
P = 0 #所有匹配的数量
TN = 0 #判不匹配并且正确的
FN = 0 #判不匹配并且错误的
N = 0 #所有不匹配的数量

objects_matching = []
P_predict_wrong_index = []
N_predict_wrong_index = []
N_predict_correct_index = []
predict_positive = []
predict_negative = []

count = 0
for i in range(len(data)):
	expression = ''
	predict = 0
	image_id = data[i]["image_id"]
	expression_list = image_id.split(' ')[2:]
	objects = data[i]["objects"]
	relations = data[i]["relations"]
	ed_image_id = 'COCO_train2014_' + image_id.split(' ')[0] + '.jpg'
	ed_element = entity_dictionary[ed_image_id][0]
	n = 0
	# print(data[i])
	misobj = []
	
	for o in objects:
		state = False
		o_s = wordnet_syn(o)
		# print(o_s)
		if len(o_s) == 0:
			continue
		for e in ed_element:
			e_s = wordnet_syn(e)
			# print(e_s)
			if len(e_s) == 0:
				continue
			for o_s_e in o_s:
				for e_s_e in e_s:
					o_e_common = wordnet_com(o_s_e,e_s_e)
					# print(o_e_common)

					if len(o_e_common) == 0:
						continue
					else:
						if o_s_e == e_s_e or o_s_e == o_e_common[0] or e_s_e == o_e_common[0]:
							# print(o_e_common)
							state = True
		if state == True:
			n += 1
		if state == False:
			misobj.append(o)
			misobj = list(set(misobj))

	if n in list(range(1,len(objects)+1)):
	# if n == len(objects):
		predict = 1
		predict_positive.append(data[i])
		# print('images_name:',image_id.split(' ')+'.jpg')
		for q in image_id.split(' ')[2:]:
			expression = expression + q + ' '
		print('expression:',expression)
		print('objects:',objects)
		print('entity_dictionary:',ed_element)
		print('subject and object matching!')

	else:
		predict = 2
		predict_negative.append(data[i])
		# print('images_name:', image_id.split(' ') + '.jpg')
		for q in image_id.split(' ')[2:]:
			expression = expression + q + ' '
		print('expression:', expression)
		print('objects:',objects)
		print('entity_dictionary:',ed_element)
		for i in range(0,len(misobj)):
			print('No '+ misobj[i] +' can be found!')
		print('subject and object mismatching!')

	lable = int(image_id.split(' ')[1])

	if predict == lable:
		right = right + 1
		# print("Predicting success!")
	else:
		wrong = wrong+1
		# print("Predicting failure!")

	if lable == 1:
		P += 1
	if lable == 2:
		N += 1 

	# 样本为匹配预测为匹配
	if predict == 1 and predict == lable:
		TP += 1
	# 样本为不匹配预测为匹配
	if predict == 1 and predict != lable:
		N_predict_wrong_index.append(image_id)
		FP += 1

	# 样本为不匹配预测为不匹配
	if predict == 2 and predict == lable:
		TN += 1
	# 样本为匹配预测为不匹配
	if predict == 2 and predict != lable:
		P_predict_wrong_index.append(image_id)
		FN += 1

	# count += 1
	# if count == 60:
	# 	break

# print(predict_wrong)
print(P,TP,FN)
print(N,TN,FP)

print('Positive recall: ',TP/P,'Positive precision: ',TP/(TP+FP))
print('Negative recall: ',TN/N,'Negative precision: ',TN/(TN+FN))
print('accuracy: ',right/(right+wrong))

# path3 = 'E:/程序总结/matching detection/model/relationship_matching_detection/predict_positive_test.json'
# with open(path3,'w') as f:
# 	json.dump(predict_positive,f)

# path4 = 'E:/程序总结/matching detection/model/relationship_matching_detection/predict_negative_test.json'
# with open(path4,'w') as f:
# 	json.dump(predict_negative,f)

# print('save success')


# print()

# total_number = len(data)
# print('total_number:',total_number)
# print('right_number:',right)
# # print('wrong_number:',wrong)
# print('accuracy:',right/total_number)

# #relevanct precision
# ReP = TP/(TP+FP)
# # relevant recall
# Rerecall = TP/Re
# # irrelevanct precision
# IrP = TN/(TN+FN)
# # irrelevant recall
# Irrecall = TN/Ir

# print('relevanct precision:',ReP,'relevant recall:',Rerecall)
# print('irrelevanct precision:',IrP,'irrelevant recall:',Irrecall)


# TN = TN+78
# FP = FP-78
# right = right+78

# #relevanct precision
# ReP = TP/(TP+FP)
# # relevant recall
# Rerecall = TP/Re
# # irrelevanct precision
# IrP = TN/(TN+FN)
# # irrelevant recall
# Irrecall = TN/Ir

# print('accuracy:',right/total_number)
# print('relevanct precision:',ReP,'relevant recall:',Rerecall)
# print('irrelevanct precision:',IrP,'irrelevant recall:',Irrecall)

# print(len(object_predict_wrong))

# pre_wro_path = '/home/cuienjie/erd/data/pre_wrong/object_predict_wrong.json'

# with open(pre_wro_path,'w') as f:
# 	json.dump(object_predict_wrong,f)
# print('save success')

# #relevanct precision
# RePr = TP/(TP+(FP-3487))
# # relevant recall
# Rerecallr = TP/Re
# # irrelevanct precision
# IrPr = (TN+3487)/(TN+FN)
# # irrelevant recall
# Irrecallr = (TN+3487)/Ir
# print('relevanct precision O+R:',RePr,'relevant recall O+R:',Rerecallr)
# print('irrelevanct precision O+R:',IrPr,'relevant recall O+R:',Irrecallr)

# print('accuracy:',(right+3487)/total_number)



# 保存object_relevant
# or_save_path = 'E:/Ubuntu_Share/Question_Relevance_Detection/VTFQ/relation_detection/objects_matching.json'
# with open(or_save_path,'w') as f:
#     json.dump(objects_matching,f)
# print('object_relevant save success!')

# 保存train_object_relevant
# m = 0
# train_o_relevant_r = []
# for k in objects_matching:
#     if len(k["relations"]) == 3:
#         m += 1
#         train_o_relevant_r.append(k)
# print(m)
# train_or_save_path = 'E:/Ubuntu_Share/Question_Relevance_Detection/VTFQ/relation_detection/train_o_relevant_r.json'
# with open(train_or_save_path,'w') as f:
#     json.dump(train_o_relevant_r,f)
# print('train_o_relevant_r save success!')

# 保存test_object_relevant
# m = 0
# test_o_relevant_r = []
# for k in objects_matching:
#     if len(k["relations"]) == 3:
#         m += 1
#         test_o_relevant_r.append(k)
# print(m)
# test_or_save_path = 'E:/Ubuntu_Share/Question_Relevance_Detection/VTFQ/relation_detection/test_o_relevant_r.json'
# with open(test_or_save_path,'w') as f:
#     json.dump(test_o_relevant_r,f)
# print('test_o_relevant_r save success!')