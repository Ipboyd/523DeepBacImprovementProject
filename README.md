**INTRODUCTION:**
This effort is an update on the 2022 DeepBacs paper, specifically in regards to improving cell recall in dense, high contrast reagions. Data used in this effort can be found in the Data folder and is labeld by the figures in the original DeepBacs paper. In this work we employ a Convolutional Neural network (CNN) for pre-task training in order to create sets of features that we can use for further training. Our segmentation model is a transformer that uses the pre-task representation and the original image in order to segement the image. Our target metric is to improve upon StarDist2D's generalist model's recall or more genearlly imrpove upon the methodologies used in DeepBacs to segement cells better in high contrast, densely populated environments.



**Instructions:**
In order to corroborate our primary results, one should only have to install the dependencies and then run the respective ipynb scripts. In order to install dependences please use !pip insall {library} in order to do so in your terminal

Dependencies:
1. numpy
2. matlplotlib
3. random
4. skimage
5. torch
6. PIL
7. os
8. einops
9. cv2

Our primary results can be seen by running Transformer_with_CNN.ipynb. This notebook trains and tests on S. Aureus dataset stored on the SCC and available at https://doi.org/10.5281/zenodo.5550933. The input is the raw image transformed into grayscale and the output are individual instance masks for each bacteria in the image. We also output the sum accross all the instance masks, which is essentially a binary mask between the target objects and the background.

![image](https://github.com/user-attachments/assets/b743e9aa-56f0-4233-bd0b-d6e2f4aae666)

In order to get the bounding box results that we used to calculate recall and count please use TransformerCNN_masks.ipynb. 
In order to get the results from the augmented dataset addition please use the script data_augmentation.ipynb

**Work Log:**

-10/10/2024 - Uploaded StarDist Demo (cited in proposal) --> Using this as intial anlysis and a place to start improving classification performance (Isaac)

-11/19/2024 - Added Transformer framework Class (Omar)

-11/20/2024 - Added Generalist StarDist2D model after re-analyizing it performance on DeepBac's reported results. Added Data used in generalist model under "Data" (Isaac)

-11/20/2024 - Added the CNN framework class (Mohi)

-12/06/2024 - Added pretrained resnet for CNN (Mohi/Berk)

-12/07/2024 - Added v1 of transformer (Isaac/Omar)

-12/07 -> 12/15 - Added several different version of transformer (Everyone)

-12/15/2024 - Finalized model selection and cleaned Github (Everyone)
