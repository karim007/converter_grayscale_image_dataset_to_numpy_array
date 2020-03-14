from PIL import Image
import numpy as np
import os

def hot_encoding(labels, digits):
    examples = labels.shape[0]  
    labels = labels.reshape(1, examples)  

    label_new = np.eye(digits)[labels.astype('int32')] 
    label_new = label_new.T.reshape(digits, examples)  
    return label_new



def shufflle( array,m):

    shuffle_index = np.random.permutation(m)
    array = array[:, shuffle_index]

    return array

# def split_and_shuffle_training_test( data, label_new):
#     # m = data.shape[0] //2
#     m = data.shape[0] -2000
    
   
#     data_train, data_test = data[:m].T, data[m:].T

#     labels_train, labels_test = label_new[:,:m], label_new[:,m:]

#     shuffle_index = np.random.permutation(m)

#     data_train, labels_train = data_train[:, shuffle_index], labels_train[:, shuffle_index]

#     return data_train, labels_train, data_test, labels_test

def reshapping_dimension(data):
    return data.reshape((len(data),np.prod(data.shape[1:])))

data_folder="data/my_mnist"

data=[]
labels=[]
data_test=[]
labels_test=[]

classes=len(os.listdir(data_folder))
label_code = open("label_code.txt", "w+")

print("Start transformation")
i=0
for idx, d  in enumerate(os.listdir(data_folder)):

    if(os.path.isdir("{}/{}".format(data_folder,d))):
        print("Processsing {}".format(d))
        for f in os.listdir("{}/{}".format(data_folder,d)):
            if i % 100 != 0:
                img = Image.open("{}/{}/{}".format(data_folder,d,f))
                arr = np.array(img)
                data.append(arr)
                labels.append(d)            
            else:
                img = Image.open("{}/{}/{}".format(data_folder,d,f))
                arr = np.array(img)
                data_test.append(arr)
                labels_test.append(d)
            i=i+1



label_code.close()

print("End transformation")


print("Convert data to numpy arrays")
data=np.array(data)
labels=np.array(labels)

data_test=np.array(data_test)
labels_test=np.array(labels_test)

print("Reshaping data")
data=reshapping_dimension(data).T
data_test=reshapping_dimension(data_test).T
labels_hot_encoded= hot_encoding(labels, classes)
labels_test_hot_encoded= hot_encoding(labels_test, classes)

# print("Shuffle data")
# data= shufflle(data.T,data.shape[0])
# data_test= shufflle(data_test.T,data_test.shape[0])
# labels_hot_encoded=shufflle(labels_hot_encoded, labels_hot_encoded.shape[1])
# labels_test_hot_encoded=shufflle(labels_test_hot_encoded,labels_test_hot_encoded.shape[1])


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
np.save('output/training/label.npy', labels_hot_encoded)
np.save('output/training/data.npy', data)
np.save('output/test/label.npy', labels_test_hot_encoded)
np.save('output/test/data.npy', data_test)

print("End script")


