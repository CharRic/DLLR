# Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID 
This is the pytorch implementation of "Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID".
![framework.png](figs%2Fframework.png)

# Highlight
1. We propose a DLLR framework for USVI-ReID. By incorporating the CSM and CRLR algorithms, our framework can effectively establish associations between unlabeled samples across modalities, and then generates high quality pseudo labels for model training.
2. We design the WMM to assign different weights to samples for constructing memory banks, which can enhance the model's capacity to learn modality-invariant features by considering hard samples.
3. Extensive experiments on three public benchmarks demonstrate the superiority of our proposed method, outperforming state-of-the-art USVI-ReID methods and even surpassing many supervised VI-ReID methods.

# Data Preprocessing
Put SYSU-MM01 and RegDB dataset into ./data/sysu and ./data/regdb. Then obtain the segmented images for training:

```shell
python preprocess_sysu.py --gpu 0
```

# Training
We utilize a RTX3090 GPU for training.
```shell
./run_train_sysu.sh # for SYSU-MM01 
./run_train_regdb.sh # for RegDB
```


# Test

```shell
./run_test_sysu.sh # for SYSU-MM01 
./run_test_regdb.sh # for RegDB
```

The code is implemented based on [ADCA](https://github.com/yangbincv/ADCA).
