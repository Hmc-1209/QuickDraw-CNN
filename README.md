# QuickDraw-CNN
This is a project that uses CNN to determine what are the quick draw represent.
We used the dataset 'Quick, Draw! The Data' for training and testing our model.

Here is the link for the dataset : https://quickdraw.withgoogle.com/data

---
## **Members :** 
NTUT IAE - 1092b0014 / 1092b0003

## **Motivation :**  
You won't accept it if I say *' There is no exact motivates for doing this, we're doing this just because it looks fun :D'.*   
So, here is why : Since this is a dataset from the Google game 'Quick, Draw!', it's essential for most drawing games. Imagine if this could be
used in games like Gartic.io, and we can simply draw and let computers to helped us bring out labels, doesn't that sounds great? 
Moreover, we want to use mediapipe for drawing in the air, and then predict what it is. This could be used in various ways, and is a good chance for DIY.  
Already being excited about it ? Let's get started !!


## **What exactly can this project do ?**  
This project trains a model to visualize (28*28) pictures, and then can predict image we draw in the air.  
By default, the datasets are from .npy files given in this link : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap  
This will be automatically download but if it doesn't for some reason, download them and put them in 'dataset' folder.


## **How could I execute it ?**    
There are multiple files you could execute :
* `modelTraining.py`
  - For training datas, execute this code (or do it after customization it with your own data).  
  
 
* `predict.py`
  - For predicting the image we draw in the air, execute this code.
 
 
* `dataset (folder)`
  - `generateDatas.py`
    - It allows you to generate the amount and multiple types of images from quickdraw, it would automatically generate them in this folder.  
    - (Note that generating great quantity of images might take a lot of time !!)



---
## **References**   
* https://quickdraw.readthedocs.io/en/latest/  
* https://quickdraw.withgoogle.com/data