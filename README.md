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
img1
img2
img3


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

