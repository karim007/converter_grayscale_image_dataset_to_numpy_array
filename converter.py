from PIL import Image
import numpy as np
import os

def hot_encoding(labels, digits):
    examples = labels.shape[0]  
    labels = labels.reshape(1, examples)  

    label_new = np.eye(digits)[labels.astype('int32')] 
    label_new = label_new.T.reshape(digits, examples)  
    return label_new

def split_and_shuffle_training_test( data, label_new):
    m = data.shape[0] //2

    data_train, data_test = data[:m].T, data[m:].T
    labels_train, labels_test = label_new[:,:m], label_new[:,m:]

    shuffle_index = np.random.permutation(m)

    data_train, labels_train = data_train[:, shuffle_index], labels_train[:, shuffle_index]

    return data_train, labels_train, data_test, labels_test

def reshapping_dimension(data):
    return data.reshape((len(data),np.prod(data.shape[1:])))

data_folder="data/letters"

data=[]
labels=[]

classes=len(os.listdir(data_folder))
label_code = open("label_code.txt", "w+")

print("Start transformation")

for idx, d in enumerate(os.listdir(data_folder)):
    
    label_code.write("{} : {} ".format(d, idx))
    label_code.write("\n")
    print("Processsing {}".format(d))
    for f in os.listdir("{}/{}".format(data_folder,d)):
        
        img = Image.open("{}/{}/{}".format(data_folder,d,f))
        arr = np.array(img)
        data.append(arr)
        labels.append(idx)

label_code.close()

print("End transformation")


print("Reshaping suffle and split data")


data=np.array(data)
labels=np.array(labels)
data=reshapping_dimension(data)
labels_hot_encoded= hot_encoding(labels, classes)
data, labels, data_test, labels_test = split_and_shuffle_training_test(data,labels_hot_encoded)


output_folder="output"
training_folder="training"
test_folder="test"

if not os.path.exists(output_folder):
    os.makedirs("{}".format(output_folder))
if not os.path.exists("{}/{}".format(output_folder, training_folder)):
    os.makedirs("{}/{}".format(output_folder, training_folder))
if not os.path.exists("{}/{}".format(output_folder, test_folder)):
    os.makedirs("{}/{}".format(output_folder, test_folder))

print("Save numpy arrays")
np.save('output/training/label.npy', labels)
np.save('output/training/data.npy', data)
np.save('output/test/label.npy', labels_test)
np.save('output/test/data.npy', data_test)

print("End script")


