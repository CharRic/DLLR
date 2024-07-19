# Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID 
This is the pytorch implementation of "Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID".
![framework.png](figs%2Fframework.png)

# Highlight
1. We propose a dual-modality-shared learning and label refinement (DLLR) framework for unsupervised learning visible-infrared person re-identification (USL-VI-ReID). By incorporating the designed cluster similarity matching (CSM) and cluster relationship based label refinement (CRLR) algorithms, the framework effectively establishes associations between pseudo labels across modalities.
2. We design the weighted modality-shared memory (WMM) to assign different weights to samples for memory initialization, which enhances the model's capacity to learn modality-invariant features.
3. Extensive experiments on two benchmarks demonstrate the effectiveness of our proposed method, outperforming state-of-the-art USL-VI-ReID methods and surpassing many supervised VI-ReID methods.

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

# Contact

<font color=Blue>The information of author is not yet publicly available.</font>

The code is implemented based on [ADCA](https://github.com/yangbincv/ADCA).
