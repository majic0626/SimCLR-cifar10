# SimCLR on CIFAR10
---
“A Simple Framework for Contrastive Learning of VisualRepresentations” Appdenix B9

---
## Environment
* Ubuntu 16.04
* Python 3.5.2
* Pytorch 1.2.0+cu92
* CUDA 10.1
* Sigle RTX-2080Ti (11G)

---
## Algorithm & Concept
* The main target of unsupervised learning is to learn effective visual representation without human supervision.
* To acheive the goal, some pretext tasks are designed for networks to slove, and features are learn by objective functions.
* Features learned from models are transferred to downstream tasks for great performance.
* In this paper, they use examplar pretext task based on contrastive learning which have been recently shown great promise, achieving state-of-theart results in the field.

Now, we're talking about how the algorithm works.
First, consider there are 2 image one is a cat, and the other is an airplane. Cat is transformed into 2 images with different augmentation (random crop, random color distortion, etc.), so is the airplane. Those images then are converted by the model into visual features in high dimension feature space, e.g., 2048.From the designed loss function, the model decrease the distance between 2 cats (or 2 planes) while increase the distance between cats and airplane. The following visualize the process of the algorithm.

![figure1-1](https://github.com/majic83626/SimCLR-cifar10/blob/master/img/SimCLR_1.png)
![figure1-2](https://github.com/majic83626/SimCLR-cifar10/blob/master/img/simCLR_3.PNG)

---

## Note!!
* This experiment work on only single GPU, we use "accumulating loss"
  for larger batch. Please refer to this [great article](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255).

* Dataset directory structure should be like as follows.
  ```
  Suppose dataset is located in /usr/share/dataset/cifar10
  1. cifar10 contains 2 folers(train/test) and 1 file(class.txt)
  2. train is composed of 10 folder named by different classes.
  3. each "class folder" includes raw images belong to the class.
  
  cifar10
        |train
             |cat
                 |xxx.jpg
                 |xxx.jpg
             |dog
              .
              .
    
        |test
        |class.txt
  
  ```
  More detail is provided in doc, or you can download cifar10 [here](https://drive.google.com/drive/folders/1uQdcWh2wbDFzCgNiseiWjv1GwSWim0np?usp=sharing) (has been organized as the above struture)
  ---
  ## Experiment
  !!! UNDER CONSTRUCTION !!!


