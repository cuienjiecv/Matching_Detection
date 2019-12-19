import numpy as np
import datetime
import json
# -*- coding: UTF-8 -*-
class Glove():
	def __init__(self, fUrl):
		with open(fUrl) as f:
			self.word_dic = {line.split()[0]:np.asarray(line.split()[1:], dtype='float') for line in f}
   
	def consine_distance(self, word1, word2):
		return np.dot(self.word_dic[word1],self.word_dic[word2]) \
			/(np.linalg.norm(self.word_dic[word1])* np.linalg.norm(self.word_dic[word2]))

	def MostSimilarWord(self, word,TopN = 30):
		print word + ':'
		print self.word_dic[word]
		return sorted({word2:self.consine_distance(word, word2) for word2 in self.word_dic.keys()}.items(), \
			lambda x, y: cmp(x[1], y[1]), reverse= True) [1:TopN+1]
		
if __name__ == "__main__":
	print('nnn')
	starttime = datetime.datetime.now()
	model = Glove("/home/cuienjie/vrd/word_embeding/glove_py_model_load/vectors.txt")

	path = "/home/cuienjie/vrd/word_embeding/glove_py_model_load/predict_positive_test.json"
	with open(path,'r') as f:
		predict_positive = json.load(f)

	positive_relationship = []
	for i in range(len(predict_positive)):
		if len(predict_positive[i]['relations']) != 0:
			# print(predict_positive[i]['image_id'])
			if predict_positive[i]['image_id'].split(' ')[1] == '2':
				positive_relationship.append(predict_positive[i])
	print(len(positive_relationship))

	# {"image_id": "000000269557 1 person in blue vest", 
	# "objects": ["person", "vest"], 
	# "relations": ["person", "in", "vest"]}

	path2 = '/home/cuienjie/vrd/test_file/relationship_predict.json'
	with open(path2,'r') as f:
		relationship_dictionary = json.load(f)

	print(len(relationship_dictionary))
	# print(relationship_dictionary[1026])
	# print(relationship_dictionary[29])
	# 'COCO_train2014_000000181929.jpg': ['on', 'above']
	# print(relationship_dictionary['COCO_train2014_000000181929.jpg'][0])
	count = 1
	relationship_right = 0
	k = 0.85
	n = 0
	item = [1,4,7,10,13,16,19,21]

	for i in range(len(positive_relationship)):
		n += 1
		state = False
		image_name = 'COCO_train2014_' + positive_relationship[i]["image_id"].split(' ')[0] + '.jpg'
		print(image_name)

		pos_rel_ele = positive_relationship[i]['relations'][1]

		for rel_dic_ele in relationship_dictionary[image_name]:
			try:
				if model.consine_distance(pos_rel_ele.split(' ')[0], rel_dic_ele.split(' ')[0]) > k :
					state = True
			except KeyError:
				pass

		if state == True:
			predict = 1
		else:
			predict = 2

		lable = positive_relationship[i]["image_id"].split(' ')[1]

		if int(lable) == predict:
			relationship_right += 1
	
	print(relationship_right)

	P = float(35152 - relationship_right)
	TP = float(29272) 
	FN = float(5880 - relationship_right)

	N = float(34954 + relationship_right)
	TN = float(30637 + relationship_right)
	FP = float(4317)

	accuracy = (TP+TN)/(P+N)

	print 'Positive recall: ',TP/P,'Positive precision: ',TP/(TP+FP)
	print 'Negative recall: ',TN/N,'Negative precision: ',TN/(TN+FN)
	print 'accuracy: ',accuracy

