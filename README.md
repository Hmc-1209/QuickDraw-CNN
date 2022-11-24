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
That's why we want to give it a try.  

## **What exactly can this project do ?**  
This project trains a model to visualize (28*28) pictures. By default, the datasets are from .npy files given in this link : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

## **Before you start :**
Please make sure to download these files from the link given in the previous paragraph :  
`full_numpy_bitmap_` + (`cat`, `diamond`, `eye`, `ladder`, `moon`, `necklace`, `snowflake`, `sword`, `tornade`, `wine glass`) + `.npy`  
These files are the type of images we train in this project. Put them in ' dataset ' folder.  




>## **How can I execute it ?**    
>There are multiple files you could execute :
>* `modelTraining.py`
>  - For training datas, execute this code (or do it after customization it with your own data).  
>  
> 
>* `xxx.py`
>  - xxx
> 
> 
>* `dataset (folder)`
>  - multiple `.npy` files
>    - Put the files you've downloaded previously here.
>    - By default, this is the best practice to get your data, but you could also give it a try to save local images using 'generateDatas.py'.
>  - `generateDatas.py`
>    - Inside this file, it allows you to generate the amount and multiple types of images from quickdraw,  
       it would automatically generate them in this folder. (Note that generating great quantity of images might take a lot of time !!)


---
## **References**   
* https://quickdraw.readthedocs.io/en/latest/  
* https://quickdraw.withgoogle.com/data