# Introduction: 
Grabcut is an algorithm for image segmentation of background and foreground. It is considered as one of the semiautomatic image segmentation techniques since it requires user interaction for the initialization of the segmentation process. This automation of the GrabCut technique is proposed as a modification of the original semiautomatic one to eliminate the user interaction. 

## Methodology 1:
There are multiple research papers out there, discussing techniques such as unsupervised Orchard and Bouman clustering technique for the initialization phase and Maximization Algorithm for Gaussian mixture model to replace the original initialization process and achieve automatic GrabCut. 

For this project I’ll be discussing two techniques that I found to be simple and yet very effective in achieving excellent results. The first method is based on finding the largest contour from the input image. A contour can be considered as a sequence of points that define a specific object in an image. Once we found it, it can be passed as bounding Rectangle coordinates for the GrapCut Algorithm to work on, hence making it fully automatic. 

## Use case: 
This method can be used as a pre-processing technique on images to train CNN or LSTM models where entropy is quite high, and we only need to extract the main object resulting in faster training time and more accuracy. I’ll be testing this method on LRCN model with high entropy dataset and will be sharing its analysis soon. 

## Testing: 
### Input Images: (From simple to complex) 
<img src="/Test-images/test.jpg" width="240"/> <img src="/Test-images/test3.jpg" width="240"/>  <img src="/Test-images/test4.jpg" width="240"/>  <img src="/Test-images/test5.jpg" width="240"/>  <img src="/Test-images/test6.jpg" width="240"/> <img src="/Test-images/test7.jpg" width="240"/> 

### Output Results:
<img src="/contours results/test.jpg" width="240"/> <img src="/contours results/test3.jpg" width="240"/>  <img src="/contours results/test4.jpg" width="240"/>  <img src="/contours results/test5.jpg" width="240"/>  <img src="/contours results/test6.jpg" width="240"/> <img src="/contours results/test7.jpg" width="240"/> 

## Weakness:
However, this method is not Foolproof, lets take an example in which suppose what if we have an image in which our foreground that we want to extract covers up all the image itself. In this case the method will try to find the largest contour within the object that we want to extract resulting in wrong output.

### Example: 
Let’s suppose we want to extract, the stop the board from the background using the same methodology discussed above.  It will only extract “P” which isn’t out desired outcome. The reason behind this is that “P” came out to be the largest contour within the sign board which covers up all the image. Although these cases are rare, but they do exist which can be resolved by applying some limitation such as camera distance when taking the image or by doing modifications to algo itself responsible of getting contours. This takes us to the second technique which works on different methodology to solve this. 

<img src="/Test-images/test2.jpg" width="240"/> <img src="/contours results/test2.jpg" width="240"/>

## Methodology 2:

This methodology works on cascade classifier to segment the desired objects from the image and passing it’s coordinates as bounding Rectangle for the GrapCut in fine-tuning the results. Cascade classifiers are trained using several positive images and arbitrary negative images. OpenCV contains several pretrained cascading classifiers used in image processing to detect objects. Moreover, there are multiple pretrained cascade classifiers on internet that you can find to work around. 

## Testing:
For this method ill be testing the same image which was our largest contour method failed to segment. 

<img src="/Test-images/test2.jpg" width="240"/> <img src="/cascade results/test2.jpg" width="240"/>

## Room for improvement
You are always welcome to imporve on this. The code in general is very simple and easy to understand.

# Installation: 
1) Clone the repo.
2) To install opencv 
``` pip install opencv-python ```
3) To test 1st Methodology
``` python main.py "images directory" ```
4) To test 2nd methodology
``` python cascade.py "images directory" ```
