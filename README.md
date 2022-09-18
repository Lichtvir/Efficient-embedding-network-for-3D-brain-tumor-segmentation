# Efficient-embedding-network-for-3D-brain-tumor-segmentation

This is the source code for the paper [Efficient embedding network for 3D brain tumor segmentation](https://link.springer.com/chapter/10.1007/978-3-030-72084-1_23).

The proposed segmentation approach follows a convolutional encoder-decoder architecture. 
It is built from an asymmetrically large encoder to extract image features and a smaller decoder to reconstruct segmentation masks.

![model](https://user-images.githubusercontent.com/83643719/190924740-082acc0f-b7e3-4665-8634-8b02139cb2b4.PNG)

The multimodal Brain Tumor Segmentation (BraTS) challenge aims at encouraging the development of state-of-the-art methods for the segmentation of brain tumors by providing a large 3D MRI dataset of annotated LGG and HGG. The BraTS 2020 training dataset include 369 cases (293 HGG and 76 LGG), each with 4 modalities describing: native (T1), post-contrast T1-weighted (T1Gd), T2-weighted (T2), and T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, which were acquired with different clinical protocols and various MRI scanners from multiple (n=19) institutions. Each tumor was segmented into edema, necrosis and non-enhancing tumor, and active/enhancing tumor. Annotations were combined into 3 nested sub-regions: Whole Tumor (WT), Tumor
Core (TC) and Enhancing Tumor (ET).

![train_samples](https://user-images.githubusercontent.com/83643719/190926963-fa19941b-3ab8-4807-9755-1b9f1b8c5882.png)



The high/low intensity of the tumor parts indicates that the features of the network are focused on the detection of the tumor.
Results of the output of the passage of one volume is shown below.

![vizu](https://user-images.githubusercontent.com/83643719/190928869-d630bc8a-4e76-45a1-ace2-00d4a59b25bd.png)

The proposed network architecture is trained with centered cropped data of size 192 × 160 × 108 voxels, ensuring that the useful content of each slice remains within the boundaries of the cropped area, training was made on BraTS2020 dataset. The batch size is set to 1. Training has been done using the Nadam optimizer with an initial learning rate of 0.0001 reduced by a factor of 10 whenever the loss do not improve for 50 epochs. 

The tensorflow/keras implementation of the network is availale [HERE](https://github.com/Lichtvir/Efficient-embedding-network-for-3D-brain-tumor-segmentation/blob/main/model.py).
