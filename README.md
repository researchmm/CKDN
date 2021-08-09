# CKDN
The official implementation of the ICCV2021 paper "Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment"

![screenshot 173](https://user-images.githubusercontent.com/35843017/128686858-534ac4c1-8556-4999-a2cb-695119251e78.jpg)

Our trained model can be found in [Model](https://drive.google.com/file/d/13HzxPsRWdPC_CAsbvgMgU5IG_TKPVbNe/view?usp=sharing)

The PIPAL dataset with our matched degraded images can be downloaded in [Data] (https://drive.google.com/file/d/13HzxPsRWdPC_CAsbvgMgU5IG_TKPVbNe/view?usp=sharing)

To train the model, please run:

>bash train.sh

To evaluate the model, please run:

>bash val.sh

To predict the quality score for an image/folder, please:

>1. put degraded images into 'data_folder/degraded' and restored images into 'data_folder/restored' (with the same file name).
>2. run: bash predict_score.sh



## Citation

    @article{zheng2021learning,
      title={Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment},
      author={Zheng, Heliang and Fu, Jianlong and Zeng, Yanhong and Zha, Zheng-Jun and Luo, Jiebo},
      journal={ICCV},
      year={2021}
    }

