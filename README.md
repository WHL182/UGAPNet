Dependencies 

⚫ Python 3.8

⚫ PyTorch 1.7.0

⚫ cuda 11.0

⚫ torchvision 0.8.1

⚫ tensorboardX 2.14

Datasets 

⚫ PASCAL-5i: VOC2012 + SBD

⚫ COCO-20i: COCO2014

⚫ Download the data lists (.txt files) and put them into the UGAPNet/lists
directory.

⚫ Run util/get_mulway_base_data.py to generate base annotations for stage1, or 
directly use the trained weights.

Models 

⚫ Download the pre-trained backbones from here and put them into the
UGAPNet /initmodel directory.

⚫ Download BAM model base learners from OneDrive and put them under
initmodel/PSPNet.

⚫ We provide some trained UGAPNet models for performance evaluation.
Backbone: VGG16 & ResNet50; Dataset: PASCAL-5i & COCO-20i; Setting: 
1-shot & 5-shot. All trained models will be obtained after acceptance of the 
paper.

References

⚫ This repo is mainly built based on PFENet, BAM, and MSANet. Thanks for 
their great work!

Scripts

⚫ Change configuration via the .yaml files in UGAPNet /config, then run the 
test.py file for testing
