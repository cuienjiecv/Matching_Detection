import os
# # for file_name in os.listdir(r"E:/Ubuntu_Share/Question_Relevance_Detection/VTFQ/images/VTFQ_images"):
# #     print(file_name)
# a = {}
# b = []
# c = 10
# for i in range(10):
#     print(i)
#     b.append(i)
# a[c] = b
# print(b)
# print(a)
# d = [11,11,12]
# y = list(set(d))
# print(y)
a = [[1,2,3],[2,3,4]]
b = [5,6]
c = {}
c['image'] = []
c['image'].append(a)
c['image'].append(b)
print(c)

dir_path = "/home/cuienjie/erd/object_detect/Mask_RCNN-master/images"
for file_name in os.listdir(dir_path):
	print(file_name)
	# image = skimage.io.imread(dir_path+file_name)
 #    print(file_name,' the images is detecting')
