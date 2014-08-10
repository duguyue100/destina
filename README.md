Destina
=======

##Introduction##

A DeSTIN implementation.

This project is developed under Ubuntu 12.04.3 and Mac OS X.

##To-do List##

1. Information filter based sparse coding.
2. Fast inverse (optional)
3. Test on Gabor wavelet dictionary and natural image with sparse coding. [DONE 20140422]
4. Clustering algorithm for upper layer. [DONE changed to sparse autoencoder]
5. Test on second layer learned representation.
6. Centroids/Dictionary visualisation.
7. Signal composition and decomposition [DONE 20140425].
8. General mapping between layers [DONE 20140504].
9. Performance evaluation.
10. Memory leaking. [DONE 20140622]
11. Interface for CIFAR image [DONE 20140621].
12. Some visualisation tools for image dataset. [DONE]
13. Experiment result saving functions (needed to rescale to `uchar` space).
14. Experiment with CIFAR images [DONE 20140622 (UPDATED IN 20140729)].

##Updates##

1. Gabor wavelet dictionary design [2014-04-22]
2. Image whitening updated [2014-04-24]
3. Information filter based sparse coding and test updated [2014-04-25 (undone)]
4. Main components of DeSTIN network are sketched [2014-04-27]
5. Sparse Autoencoder is updated [2014-06-14]
6. General Refinement of DeSTIN architecture is updated [2014-06-21]
7. CIFAR classification example sketch [2014-06-22]
8. Revised contrast normalisation and whitening [2014-06-23]
9. CIFAR classification on complete set [2014-07-29]
10. Nao action learning update (up to 9 scenes) [2014-08-04]
11. Destina Python support by SWIG [2014-08-09]

##Setup##

1. Clone the whole project (Matlab codes also provide some insight of the project)

2. `cd` in `destina` folder and configure with `cmake`

   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

3. Some demonstration programs can be found in `src`, and executable can be found in `build/bin`.

4. If you want to update the program (fork and clone from your own account), you can simply use `update-all.sh`, run following code in terminal:
   ```
   $ sh update-all.sh
   ```

5. __How to set up CIFAR-10 test__: download CIFAR-10 dataset and unzip it under `destina` folder. Then after you build the project, under `build/bin` you should run `classify`, then if nothing wrong, your test should go well. All the details are specified in `src/classify_test.cpp`. You can change line 20 to specify which dataset you are using. Currently, I just wrote up a example on one training dataset, you can do more on the rest by modify the program. To ensure your are on the right track, here is the folder structure  from my project
   ```
   /destina
       /build
       /cifar-10-batches-bin
       CMakeLists.txt
       /include
       LICENSE
       README.md
       /resources
       /src
       /Python
       /swig_destina_common.i
       update-all.sh
   ```

##Notes##

1. Example of Gabor Wavelet dictionary

   ![Gabor Wavelet Dictionary](/resources/gabor_dictionary_32_32.png)

2. Example of Whitened Image

   ![Whitened Image](/resources/whitening.png)

3. In C++ implementation, can only learn some representations roughly. However the performance is not acceptable, still need to figure out it's dictionary's problem or implementation's problem. Furthermore, the speed is not acceptable for now.

4. Example of reconstruction image with sparse autoencoder

   ![Sparse Autoencoder Reconstruction](/resources/sparse_autoencoder_reconstruction.png)

5. In this implementation, to ensure that DeSTIN is started from a better place, I employed a bottom-up batch training to train the autoencoder in each layer with a batch of images. And after this pre-training, the training stage will be performed as usual and the  speed is quite fast. __Noted that pre-training stage would take a lot of times, so it needs to consider another way to speedup.__ [2014-06-22]

6. Image Reconstruction indicated that this implementation is performed OK on average. So that if we only employ first layer's learning result, it should still give us state-of-art performance with a plain autoencoder. [2014-06-22]

##Contacts##

Hu Yuhuang

Advanced Robotics Lab

Department of Artificial Intelligence

Faculty of Computer Science & IT

University of Malaya

duguyue100@gmail.com
