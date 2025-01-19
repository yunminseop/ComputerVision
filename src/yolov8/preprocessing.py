from glob import glob
from sklearn.model_selection import train_test_split

images_path = "/home/ms/my_yolo_data/dataset/images/"
labels_path = "/home/ms/my_yolo_data/dataset/labels/"

img_list = glob(images_path+"*.jpg")
label_list = glob(labels_path + "*.txt")
print(len(img_list), len(label_list))


train_img_list, val_img_list = train_test_split(img_list, test_size = 0.3, random_state = 200)
print(len(train_img_list), len(val_img_list))
    

# save images list for yaml file.
with open('/home/ms/my_yolo_data/train.txt','w') as f:
    f.write('\n'.join(train_img_list)+'\n')
with open('/home/ms/my_yolo_data/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list)+'\n')