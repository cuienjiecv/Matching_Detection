from nltk.corpus import wordnet as wn

objects = ['laptop','animal']
ed_element = ['computer','dog','cat']

# objects = ['back']
# ed_element = ['person','horse', 'shirt']

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

print(wordnet_syn('dog'))
print(wordnet_syn('cat'))

# print(wordnet_syn('dog')[].lowest_common_hypernyms(wordnet_syn('cat')[0]))

# print(wn.synset('dog.n.03').lowest_common_hypernyms(wn.synset('cat.n.01')))

n = 0
for o in objects:
	o_s = wordnet_syn(o)
	if len(o_s) == 0:
		continue
	state = False
	for e in ed_element:
		e_s = wordnet_syn(e)
		if len(e_s) == 0:
			continue
		for o_s_e in o_s:
			for e_s_e in e_s:
				o_e_common = wordnet_com(o_s_e,e_s_e)
				if len(o_e_common) == 0:
					continue
				if o_s_e == e_s_e or o_s_e == o_e_common[0] or e_s_e == o_e_common[0]:
					# 只要发生这个动作 state = True
					# print(o,e)
					state = True
					# print(o_s_e,e_s_e)
					# print(o_e_common[0],e_s_e)
	if state == False:
		print('There is no',o)
	if state == True:
		n += 1
		print('There is a',o)

# print(wn.synset('cow.n.03').definition())
print(n)


# image_id = '000000069167 1 boy on horse'
# for w in image_id.split(' ')[2:]:
# 	print(w)
# expression = 'computer on the table'
# print(expression.split(' '))
		# if m in list(range(1,len(expression.split(' ')[2:])+1)):
		# 	predict = 1
		# 	for q in image_id.split(' ')[2:]:
		# 		expression = expression + q + ' '
		# else:
		# 	predict = 2
		# 	for q in image_id.split(' ')[2:]:
		# 		expression = expression + q + ' '