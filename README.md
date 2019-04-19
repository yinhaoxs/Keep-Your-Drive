# Arcface_KYD

Pytorch0.4.1 codes for Arcface_KYD

## 1. Intro

- This repo is a implementation of FicialRiskNet
- For models, including the pytorch implementation of the backbone modules of Arcface and MobileFacenet. 
- RankNet, Wo introduce a implementation of RankNet for driving risk

## 2. Pretrained Models & Performance

[IR-SE50 @ BaiduNetdisk](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ), [IR-SE50 @ Onedrive](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC)

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9952 | 0.9962    | 0.9504    | 0.9622      | 0.9557   | 0.9107   | 0.9386     |

[Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg), [Mobilefacenet @ OneDrive](https://1drv.ms/u/s!AhMqVPD44cDOhkSMHodSH4rhfb5u)

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9918 | 0.9891    | 0.8986    | 0.9347      | 0.9402   | 0.866    | 0.9100     |

## 3. How to use

- clone

  ```
  ```
  git clone https://github.com/yinhaoxss/Arcface_Risk.git

  ```
  ```

### 3.1 Data Preparation(labels):

##### 3.1.1 Preparation FicialRisk faces

*** FicialRisk Faces structure: .csv file(pd file) ***

| face_picture(images path) | loss_ratio  | 
| ------------------------- | ----------- | 
| /Users/yinhao_x/kyd_1.jpg | 0.8986      |
...
| /Users/yinhao_x/kyd_m.jpg | 1.9804      |

##### 3.1.2 Preparation LFW faces

*** LFW faces structure: .csv file(pd file) ***

| face_picture(images1 path) | face_picture(images2 path)  | loss_ratio  | 
| -------------------------  | --------------------------  | ----------- | 
| /Users/yinhao_x/lfw_1.jpg  | /Users/yinhao_x/lfw_2.jpg   | 0.8986      |
...
| /Users/yinhao_x/lfw_m.jpg  | /Users/yinhao_x/lfw_n.jpg   | 1.9804      |

### 3.2 Data Preparation(images):

##### 3.2.1 Detection and align (112 * 112)

```
​```
python mtcnn_pytorch/mtcnn.py  or python dlib/align.py

​```
```
### 3.3 Training:

```
​```
python train_kydnet.py

​```
```
### 3.4 testing:

```
​```
python test_kydnet.py

​```
```

## 4 evaluating indictor

### 4.1 Loss Function

#### 4.1.1 MarginRankingLoss or soft_BCELOSS or tweedi loss

*** MarginRankingLoss ***
.. math::
        \text{loss}(x, y) = \max(0, -y * (x1 - x2) + \text{margin})


#### 4.2 Calculate Gini

```
​```
python utils/gini.py

​```
```

#### 4.3 Calculate Std
```
​```
python utils/score.py
python data/dataread.py(compute_std)

​```
```



















