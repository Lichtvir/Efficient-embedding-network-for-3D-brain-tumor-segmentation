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

📦 Model Availability 

The tensorflow/keras implementation of the network is availale [HERE](https://github.com/Lichtvir/Efficient-embedding-network-for-3D-brain-tumor-segmentation/blob/main/model.py).


📝 Citing
```
@incollection{Messaoudi2021,
  doi = {10.1007/978-3-030-72084-1_23},
  url = {https://doi.org/10.1007/978-3-030-72084-1_23},
  year = {2021},
  publisher = {Springer International Publishing},
  pages = {252--262},
  author = {Hicham Messaoudi and Ahror Belaid and Mohamed Lamine Allaoui and Ahcene Zetout and Mohand Said Allili and Souhil Tliba and Douraied Ben Salem and Pierre-Henri Conze},
  title = {Efficient Embedding Network for 3D Brain Tumor Segmentation},
  booktitle = {Brainlesion: Glioma,  Multiple Sclerosis,  Stroke and Traumatic Brain Injuries}
}
```

📚 References

1. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, and J. Kirbyet al. Segmentation labels and radiomic features for the pre-operative scans
of the tcga-gbm collection. The Cancer Imaging Archive, 2017. DOI:10.7937/K9/TCIA.2017.KLXWJJ1Q.

2. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, and J. Kirbyet al. Segmentation labels and radiomic features for the pre-operative scans
of the tcga-gbm collection. The Cancer Imaging Archive, 2017. DOI:10.7937/K9/TCIA.2017.GJQ7R0EF.

3. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, and J.S. Kirby et al. Advancing the cancer genome atlas glioma mri collections with expert segmentation labels and radiomic features. Nature Scientific Data, page 4:170117, 2017. DOI:10.1038/sdata.2017.117.

4. S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, and A. Crimi et al. Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the brats challenge. arXiv preprintarXiv:1811.02629, 2018.

5. Stefan Bauer, Roland Wiest, Lutz-P Nolte, and Mauricio Reyes. A survey of mribased medical image analysis for brain tumor studies. Phys Med Biol., 58(13):R97–R129, 2013. DOI: 10.1088/0031-9155/58/13/R97.

6. Pierre-Henri Conze, Sylvain Brochard, Val´erie Burdin, Frances T.Sheehan,and Christelle Pons. Healthy versus pathological learning transferability
in shoulder muscle MRI segmentation using deep convolutional encoderdecoders. Comput. Med. Imaging Graph., 83:101733, 2020. DOI: 10.1016/j.compmedimag.2020.101733.

7. Pierre-Henri Conze, Ali Emre Kavur, Emilie Cornec-Le Gall, Naciye Sinem Gezer,Yannick Le Meur, M. Alper Selver, and Fran¸cois Rousseau. Abdominal multi-organsegmentation with cascaded convolutional and adversarial deep networks. arXivpreprint arXiv:2001.09521, 2020.

8. B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, and J. Kirby et al. The multimodal brain tumor image segmentation benchmark
(BRATS). IEEE Transactions on Medical Imaging, 34(10):1993–2024, 2015. DOI:10.1109/TMI.2014.2377694.

9. Andriy Myronenko. 3D MRI brain tumor segmentation using autoencoder regularization. BrainLes@MICCAI, 2:311–320, 2018. DOI:10.1007/978-3-030-11726-928.

10. O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical
Image Computing and Computer-Assisted Intervention, pages 234–241, 2015.10.1007/978-3-319-24574-4 28.

11. K. Souadih, A. Belaid, D. Ben Salem, and P. H. Conze. Automatic forensic identification using 3D sphenoid sinus segmentation and deep characterization. Med.Biol. Eng. Comput., 58:291–306, 2020. DOI: 10.1007/s11517-019-02050-6.

12. Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings of Machine Learning Research, editor, 36th International Conference on Machine Learning (ICML), volume 97, pages 10691–10700, Long Beach, California, USA, 2019.

13. Minh H. Vu, Guus Grimbergen, Tufve Nyholm, and Tommy L¨ofstedt. Evaluation of multi-slice inputs to convolutional neural networks for medical image segmentation. arXiv preprint arXiv:1912.09287, 2019.

14. Y. Wu and K. He. Group normalization. In In: European Conference on Computer Vision (ECCV), 2018. DOI: 10.1007/s11263-019-01198-w.

15. R. Zaouche, A. Belaid, S. Aloui, B. Solaiman, L. Lecornu, D. Ben Salem, and S. Tliba. Semi-automatic method for low-grade gliomas segmentation in magnetic resonance imaging. IRBM, 39(2):116–128, 2018. DOI: 10.1016/j.irbm.2018.01.004.
