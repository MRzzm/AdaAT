# Adaptive Affine Transformation: A Simple and Effective Operation for Spatial Misaligned Image Generation. (accepted in MM2022)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f2a394c4a2dd4e738caffa9c2c2b5f91.png#pic_center)

Paper    [Demo video](https://www.youtube.com/watch?v=OJtGbsMWZ3I)    Supplementary materials

# Face
I'm preparing face related codes.
# Person image generation
### Download resources (pretrained model etc.) 
Download resources in [Google drive](https://drive.google.com/drive/folders/1g97_HTqCex7QKEofYklNtCfvPAKxu3bP?usp=sharing), including: 

+ **person_epoch_30.pth:** Pretrained model on deep fashion dataset stopped in 30 epoch.
+ **person_epoch_40.pth:** Pretrained model on deep fashion dataset stopped in 40 epoch (has better performance than "person_epoch_30.pth").
+ **test_image_person_deepFashion_30epoch.zip:** Inference images of "person_epoch_30.pth" on deep fashion test data for convenient comparisons.
+ **test_image_person_deepFashion_40epoch.zip:** Inference images of "person_epoch_40.pth" on deep fashion test data for convenient comparisons.
+ **example_person_source_img.jpg:** Source example image for person image generation.
+ **example_person_souce_kp.txt:** Source example key points for person image generation.
+ **example_person_target_kp.txt:** Target example key points for person image generation.
+ **example_person_inference_img.jpg:** Inference example image for person image generation.
+ **fasion_train_data.json:**   Training json file of deep fashion dataset.\

### Train on deep fashion dataset
1. Download deep fashion dataset from [here](http://disi.unitn.it/~hao.tang/uploads/datasets/SelectionGAN/fashion_data.tar.gz). We use the dataset as same as in [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN).
2. Unzip the dataset.
3. run 
```python 
python train_person_image.py --train_data=./assert/fasion_train_data.json --train_img_dir=./deepFashion/fashion_data/train
```
### Inference
To inference one person image from one source person image, source key points and target key points, run
```python 
python inference_person_image.py --inference_model_path=./assert/person_epoch_30.pth --source_img_path=./assert/example_person_source_img.jpg --source_kp_path=./assert/example_person_souce_kp.txt --target_kp_path=./assert/example_person_target_kp.txt --res_person_path=./assert/example_person_inference_img.jpg
```

### Compute metrics
To compute the metrics of SSIM and LIPIS on deep fashion test data, run
```python 
python metrics_person.py --inference_img_dir --real_img_dir=./deepFashion/fashion_data/test --task_type=person
```
# Citation
If you use AdaAT operator in your work, please cite

>@inproceedings{zhang2022adaptive,
  title={Adaptive Affine Transformation: A Simple and Effective Operation for Spatial Misaligned Image Generation},
  author={Zhang, Zhimeng and Ding, Yu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1167--1176},
  year={2022}
}
# Acknowledgement
The basic modules are borrowed from [first-order-model](https://github.com/AliaksandrSiarohin/first-order-model), thanks for their contributions.