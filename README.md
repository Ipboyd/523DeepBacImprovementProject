**INTRODUCTION:**
This effort is an update on the 2022 DeepBacs paper, specifically in regards to improving cell recall in dense, high contrast reagions. Data used in this effort can be found in the Data folder and is labeld by the figures in the original DeepBacs paper. In this work we employ a Convolutional Neural network (CNN) for pre-task training in order to create sets of features that we can use for further training. Our segmentation model is a transformer that uses the pre-task representation and the original image in order to segement the image. Our target metric is to improve upon StarDist2D's generalist model's recall or more genearlly imrpove upon the methodologies used in DeepBacs to segement cells better in high contrast, densely populated environments.


**ChangeLog:**

-10/10/2024 - Uploaded StarDist Demo (cited in propsal) --> Using this as intial anlysis and a place to start imroving classification preformance (Isaac)

-11/19/2024 - Added Transformer framework Class (Omar)

-11/20/2024 - Added Generalist StarDist2D model after re-analyizing it performance on DeepBac's reported results. Added Data used in generalist model under "Data" (Isaac)

-11/20/2024 - Added the CNN framework class (Mohi)

-12/06/2024 - Added pretrained resnet for CNN (Mohi/Berk)

-12/07/2024 - Added v1 of transformer (Isaac/Omar)
