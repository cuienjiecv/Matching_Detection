from nltk.corpus import wordnet as wn

def wordnet_syn(word):
	try:
		syn = wn.synsets(word)
	except AttributeError:
		syn = []
	return syn

# def wordnet_com(syn1,syn2):
# 	try:
# 		syn_common = syn1.lowest_common_hypernyms(syn2)
# 	except AttributeError:
# 		syn_common = []
# 	return syn_common

def wordnet_com(syn1,syn2):
	syn_common = syn1.lowest_common_hypernyms(syn2)
	return syn_common


print(str(wordnet_syn('cat')[0]).split('(')[1].split(')')[0])

print(wordnet_syn('dog'))


# print(wordnet_com(Synset('dog.n.01'),Synset('dog.n.01')))
