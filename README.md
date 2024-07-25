# Double-well-Net

This is the Pytorch code for "Double-well Net for image segmentation" by Hao Liu, Jun Liu, Raymond Chan, and Xue-Cheng Tai.

If this code is useful, pleae cite our paper
arXiv preprint arXiv:2401.00456.

Copyright (c) Hao Liu (haoliu AT hkbu.edu.hk)
Department of Mathematics,
Hong Kong Baptist University,
Kowloon, Hong Kong
https://www.math.hkbu.edu.hk/~haoliu/

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Software is provided "as is", without warranty of any kind.

Datasets used in our paper:<br />
MSRA10K: https://mmcheng.net/msra10k/<br />
ECSSD: http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html<br />
RITE: https://medicine.uiowa.edu/eye/rite-dataset

**Demo**:<br />
train.py trains the model.<br />
The dataloader in this file is for the MSRA10K dataset.

After changing the data dictionaries in train.py to your own ones, you can run train_new.py to train the model.


