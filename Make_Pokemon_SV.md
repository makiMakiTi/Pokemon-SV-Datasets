# Make Pokemon SV Datasets

```python
import json
import pprint
from PIL import Image, ImageFilter
from PIL import ImageDraw

import glob
import re
import os
from loguru import logger
from tqdm import tqdm
import numpy as np

import cv2
import sys
import shutil
```

## Setting param

データセットのパスや動画のフォルダなどのパラメーターを設定します．


```python
capture_dir = "capture"
capture_video_dir = "video"
capture_image_dir = "image"

capture_video_path = capture_dir + "/" + capture_video_dir
capture_image_path = capture_dir + "/" + capture_image_dir

root_path = "/home"

diff_image_th = 1000
save_freq     = 4

datasets_dir  = "datasets"
datasets_ver  = "v0"
datasets_path = datasets_dir + "/" + datasets_ver

anotate_full = "datasets/v0/result.json"
anotate_full_repath = "datasets/v0/result_repath.json"

anotate_train_name = "pokemon_sv_train.json"
anotate_train_path = datasets_path + "/" + anotate_train_name
anotate_valid_name = "pokemon_sv_valid.json"
anotate_valid_path = datasets_path + "/" + anotate_valid_name

image_full_dir  = "images"
image_train_dir = "train2017"
image_valid_dir = "val2017"
```


```python
%cd $root_path
```

    /home



```python
!ls
```

    Dockerfile  README.md  capture	datasets  docker-compose.yml  notebook	utils


## キャプチャー動画の分解

キャプチャーした動画を分解して画像に変換します．

変換のない静止した状態の画像はスキップした上で，`save_freq`フレームごとに画像を保存します．


### キャプチャー動画のリストを取得


```python
glob_path = capture_video_path + "/*.mp4"
video_list = glob.glob(glob_path, recursive=True)
pprint.pprint(video_list)
```

    ['capture/video/2022-11-19_11-55-09.mp4']


### 動画の分解と保存


```python
def analysis_video(video_path):
    
    video_name = video_path.split("/")[-1]
    video_single_path = capture_image_path + "/" + video_name
    logger.info("{:>20} : {}".format("video_single_path", video_single_path))
    os.makedirs(video_single_path, exist_ok=True)
       
    
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    image_id = 1
    
    while True:
        ret, frame = cap.read()

        # 読み込み可能かどうか判定
        if ret:
            logger.info("========================")
            logger.info("{:>20} : {}".format("count", count))
            
            # 0番目は pre frameに登録のみで処理はskip
            if(count==0):
                pre_frame = frame
            else:
                # 0番目以降は処理
                
                # 差分を計算
                diff_image = np.sum(np.abs(pre_frame - frame))
                logger.info("{:>20} : {}".format("diff_image", diff_image))
                
                # 閾値以上なら処理する
                if(diff_image > diff_image_th):
                    # 一定間隔で画像を保存
                    if(image_id % save_freq == 0):
                        save_image_name = "{:09d}.jpg".format(image_id)
                        save_image_path = video_single_path + "/" + save_image_name
                        logger.info("{:>20} : {}".format("save_image_path", save_image_path))
                        cv2.imwrite(save_image_path, frame)
                                                          
                    image_id += 1
                pre_frame = frame
                
                
            count += 1
        else:
            logger.info("Video Fin ...")
            break
            
        
```


```python
def video_section():
    for video_path in video_list:
        logger.info("{:>20} : {}".format("video_path", video_path))
        analysis_video(video_path)
```


```python
#video_section()
```

## 画像をアノテーション

こちらのアノテーションソフトを使ってアノテーションしていきます．

https://github.com/makiMakiTi/label-studio-1.6.0

下記のコマンドにて実行可能です．

```bash
docker-compose up --build
```

## アノテーションファイルの修正

exportされたアノテーションファイル`datasets\v0\result.json`は画像のパスが`COCO`フォーマットになっていないので修正します．



読み込みます


```python
with open(anotate_full, 'rt', encoding='UTF-8') as annotations:
    result_coco = json.load(annotations)
```

パスを修正しファイル名にします．


```python
for i in range(len(result_coco["images"])):
    file_name = result_coco["images"][i]['file_name']    
    result_coco["images"][i]['file_name'] = file_name.split("/")[-1]
```

書き出します．


```python
with open(anotate_full_repath, 'wt', encoding='UTF-8') as coco:
        json.dump(result_coco, coco, indent=2, sort_keys=True)
```

## データセットの split

データセットの分割します．


```python
!python utils/cocosplit.py --having-annotations --multi-class -s 0.8 $anotate_full_repath $anotate_train_path $anotate_valid_path
```

    Saved 87 entries in datasets/v0/pokemon_sv_train.json and 22 in datasets/v0/pokemon_sv_valid.json



```python
def move_datasets_image_file(target_dir, anno_path):
    
    logger.info("{:>20} : {}".format("target_dir", target_dir))
    logger.info("{:>20} : {}".format("anno_path", anno_path))
    os.makedirs(target_dir, exist_ok=True)
    
    with open(anno_path, 'rt', encoding='UTF-8') as annotations:
        result_coco = json.load(annotations)

    for i in range(len(result_coco["images"])):
        logger.info(">>>>>>>>>>>> {:>20} : {}".format("i", i))
        
        file_name = result_coco["images"][i]['file_name']   
        logger.info("{:>20} : {}".format("file_name", file_name))
        
        source_path =  datasets_path + "/" + image_full_dir + "/" + file_name
        logger.info("{:>20} : {}".format("source_path", source_path))
        
        target_path =  target_dir + "/" + file_name
        logger.info("{:>20} : {}".format("target_path", target_path))
        
        shutil.copyfile(source_path, target_path)
        
    #pprint.pprint(result_coco)
```


```python
move_datasets_image_file(target_dir=datasets_path + "/" + image_train_dir, anno_path=anotate_train_path)
```

    2022-11-19 05:37:55.699 | INFO     | __main__:move_datasets_image_file:3 -           target_dir : datasets/v0/train2017
    2022-11-19 05:37:55.700 | INFO     | __main__:move_datasets_image_file:4 -            anno_path : datasets/v0/pokemon_sv_train.json
    2022-11-19 05:37:55.711 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 0
    2022-11-19 05:37:55.712 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 67d01f48-000000068.jpg
    2022-11-19 05:37:55.713 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/67d01f48-000000068.jpg
    2022-11-19 05:37:55.714 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/67d01f48-000000068.jpg
    2022-11-19 05:37:55.765 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 1
    2022-11-19 05:37:55.767 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 045754af-000000124.jpg
    2022-11-19 05:37:55.768 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/045754af-000000124.jpg
    2022-11-19 05:37:55.770 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/045754af-000000124.jpg
    2022-11-19 05:37:55.816 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 2
    2022-11-19 05:37:55.817 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 7bc2ad02-000000120.jpg
    2022-11-19 05:37:55.818 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/7bc2ad02-000000120.jpg
    2022-11-19 05:37:55.819 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/7bc2ad02-000000120.jpg
    2022-11-19 05:37:55.863 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 3
    2022-11-19 05:37:55.864 | INFO     | __main__:move_datasets_image_file:14 -            file_name : c374c1d1-000000116.jpg
    2022-11-19 05:37:55.865 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/c374c1d1-000000116.jpg
    2022-11-19 05:37:55.866 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/c374c1d1-000000116.jpg
    2022-11-19 05:37:55.910 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 4
    2022-11-19 05:37:55.911 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 1658a838-000000112.jpg
    2022-11-19 05:37:55.911 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/1658a838-000000112.jpg
    2022-11-19 05:37:55.912 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/1658a838-000000112.jpg
    2022-11-19 05:37:55.958 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 5
    2022-11-19 05:37:55.959 | INFO     | __main__:move_datasets_image_file:14 -            file_name : cc9038c8-000000108.jpg
    2022-11-19 05:37:55.960 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/cc9038c8-000000108.jpg
    2022-11-19 05:37:55.961 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/cc9038c8-000000108.jpg
    2022-11-19 05:37:56.005 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 6
    2022-11-19 05:37:56.006 | INFO     | __main__:move_datasets_image_file:14 -            file_name : c4def748-000000104.jpg
    2022-11-19 05:37:56.007 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/c4def748-000000104.jpg
    2022-11-19 05:37:56.008 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/c4def748-000000104.jpg
    2022-11-19 05:37:56.047 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 7
    2022-11-19 05:37:56.048 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 434f0155-000000100.jpg
    2022-11-19 05:37:56.049 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/434f0155-000000100.jpg
    2022-11-19 05:37:56.050 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/434f0155-000000100.jpg
    2022-11-19 05:37:56.085 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 8
    2022-11-19 05:37:56.086 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 3048621e-000000096.jpg
    2022-11-19 05:37:56.087 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/3048621e-000000096.jpg
    2022-11-19 05:37:56.088 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/3048621e-000000096.jpg
    2022-11-19 05:37:56.129 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 9
    2022-11-19 05:37:56.130 | INFO     | __main__:move_datasets_image_file:14 -            file_name : f4a6c0f3-000000092.jpg
    2022-11-19 05:37:56.131 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/f4a6c0f3-000000092.jpg
    2022-11-19 05:37:56.132 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/f4a6c0f3-000000092.jpg
    2022-11-19 05:37:56.182 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 10
    2022-11-19 05:37:56.183 | INFO     | __main__:move_datasets_image_file:14 -            file_name : d1d78546-000000088.jpg
    2022-11-19 05:37:56.183 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/d1d78546-000000088.jpg
    2022-11-19 05:37:56.185 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/d1d78546-000000088.jpg
    2022-11-19 05:37:56.238 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 11
    2022-11-19 05:37:56.239 | INFO     | __main__:move_datasets_image_file:14 -            file_name : d8a5a419-000000084.jpg
    2022-11-19 05:37:56.240 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/d8a5a419-000000084.jpg
    2022-11-19 05:37:56.241 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/d8a5a419-000000084.jpg
    2022-11-19 05:37:56.284 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 12
    2022-11-19 05:37:56.286 | INFO     | __main__:move_datasets_image_file:14 -            file_name : e3b48321-000000080.jpg
    2022-11-19 05:37:56.287 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/e3b48321-000000080.jpg
    2022-11-19 05:37:56.288 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/e3b48321-000000080.jpg
    2022-11-19 05:37:56.331 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 13
    2022-11-19 05:37:56.332 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 57a35916-000000076.jpg
    2022-11-19 05:37:56.333 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/57a35916-000000076.jpg
    2022-11-19 05:37:56.333 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/57a35916-000000076.jpg
    2022-11-19 05:37:56.377 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 14
    2022-11-19 05:37:56.378 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 2f99697f-000000072.jpg
    2022-11-19 05:37:56.378 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/2f99697f-000000072.jpg
    2022-11-19 05:37:56.379 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/2f99697f-000000072.jpg
    2022-11-19 05:37:56.425 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 15
    2022-11-19 05:37:56.426 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 6ae2f8d6-000000064.jpg
    2022-11-19 05:37:56.427 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/6ae2f8d6-000000064.jpg
    2022-11-19 05:37:56.428 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/6ae2f8d6-000000064.jpg
    2022-11-19 05:37:56.474 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 16
    2022-11-19 05:37:56.475 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 75f12110-000000060.jpg
    2022-11-19 05:37:56.476 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/75f12110-000000060.jpg
    2022-11-19 05:37:56.476 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/75f12110-000000060.jpg
    2022-11-19 05:37:56.523 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 17
    2022-11-19 05:37:56.524 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 933b191b-000000056.jpg
    2022-11-19 05:37:56.524 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/933b191b-000000056.jpg
    2022-11-19 05:37:56.525 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/933b191b-000000056.jpg
    2022-11-19 05:37:56.568 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 18
    2022-11-19 05:37:56.570 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 21f44e65-000000052.jpg
    2022-11-19 05:37:56.571 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/21f44e65-000000052.jpg
    2022-11-19 05:37:56.572 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/21f44e65-000000052.jpg
    2022-11-19 05:37:56.617 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 19
    2022-11-19 05:37:56.617 | INFO     | __main__:move_datasets_image_file:14 -            file_name : db0285a4-000000048.jpg
    2022-11-19 05:37:56.618 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/db0285a4-000000048.jpg
    2022-11-19 05:37:56.620 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/db0285a4-000000048.jpg
    2022-11-19 05:37:56.663 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 20
    2022-11-19 05:37:56.664 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 0bae380e-000000044.jpg
    2022-11-19 05:37:56.664 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/0bae380e-000000044.jpg
    2022-11-19 05:37:56.665 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/0bae380e-000000044.jpg
    2022-11-19 05:37:56.717 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 21
    2022-11-19 05:37:56.718 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 0a8d7fce-000000040.jpg
    2022-11-19 05:37:56.719 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/0a8d7fce-000000040.jpg
    2022-11-19 05:37:56.721 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/0a8d7fce-000000040.jpg
    2022-11-19 05:37:56.769 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 22
    2022-11-19 05:37:56.770 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 2535401d-000000036.jpg
    2022-11-19 05:37:56.771 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/2535401d-000000036.jpg
    2022-11-19 05:37:56.771 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/2535401d-000000036.jpg
    2022-11-19 05:37:56.814 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 23
    2022-11-19 05:37:56.815 | INFO     | __main__:move_datasets_image_file:14 -            file_name : c1d2cb32-000000032.jpg
    2022-11-19 05:37:56.816 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/c1d2cb32-000000032.jpg
    2022-11-19 05:37:56.816 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/c1d2cb32-000000032.jpg
    2022-11-19 05:37:56.861 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 24
    2022-11-19 05:37:56.862 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 5aae9b2d-000000028.jpg
    2022-11-19 05:37:56.862 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/5aae9b2d-000000028.jpg
    2022-11-19 05:37:56.863 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/5aae9b2d-000000028.jpg
    2022-11-19 05:37:56.914 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 25
    2022-11-19 05:37:56.915 | INFO     | __main__:move_datasets_image_file:14 -            file_name : d8910854-000000024.jpg
    2022-11-19 05:37:56.915 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/d8910854-000000024.jpg
    2022-11-19 05:37:56.916 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/d8910854-000000024.jpg
    2022-11-19 05:37:56.965 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 26
    2022-11-19 05:37:56.966 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 1e703cb1-000000020.jpg
    2022-11-19 05:37:56.966 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/1e703cb1-000000020.jpg
    2022-11-19 05:37:56.967 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/1e703cb1-000000020.jpg
    2022-11-19 05:37:57.013 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 27
    2022-11-19 05:37:57.014 | INFO     | __main__:move_datasets_image_file:14 -            file_name : ad9dd54e-000000016.jpg
    2022-11-19 05:37:57.015 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/ad9dd54e-000000016.jpg
    2022-11-19 05:37:57.015 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/ad9dd54e-000000016.jpg
    2022-11-19 05:37:57.063 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 28
    2022-11-19 05:37:57.064 | INFO     | __main__:move_datasets_image_file:14 -            file_name : b48dba86-000000012.jpg
    2022-11-19 05:37:57.065 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/b48dba86-000000012.jpg
    2022-11-19 05:37:57.066 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/b48dba86-000000012.jpg
    2022-11-19 05:37:57.110 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 29
    2022-11-19 05:37:57.111 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 44ca9f17-000000008.jpg
    2022-11-19 05:37:57.112 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/44ca9f17-000000008.jpg
    2022-11-19 05:37:57.112 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/44ca9f17-000000008.jpg
    2022-11-19 05:37:57.152 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 30
    2022-11-19 05:37:57.154 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 0d4c392e-000000004.jpg
    2022-11-19 05:37:57.155 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/0d4c392e-000000004.jpg
    2022-11-19 05:37:57.156 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/train2017/0d4c392e-000000004.jpg



```python
move_datasets_image_file(target_dir=datasets_path + "/" + image_valid_dir, anno_path=anotate_valid_path)
```

    2022-11-19 05:37:57.201 | INFO     | __main__:move_datasets_image_file:3 -           target_dir : datasets/v0/val2017
    2022-11-19 05:37:57.203 | INFO     | __main__:move_datasets_image_file:4 -            anno_path : datasets/v0/pokemon_sv_valid.json
    2022-11-19 05:37:57.214 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 0
    2022-11-19 05:37:57.215 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 67d01f48-000000068.jpg
    2022-11-19 05:37:57.216 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/67d01f48-000000068.jpg
    2022-11-19 05:37:57.218 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/67d01f48-000000068.jpg
    2022-11-19 05:37:57.260 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 1
    2022-11-19 05:37:57.261 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 045754af-000000124.jpg
    2022-11-19 05:37:57.262 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/045754af-000000124.jpg
    2022-11-19 05:37:57.263 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/045754af-000000124.jpg
    2022-11-19 05:37:57.301 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 2
    2022-11-19 05:37:57.302 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 7bc2ad02-000000120.jpg
    2022-11-19 05:37:57.303 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/7bc2ad02-000000120.jpg
    2022-11-19 05:37:57.304 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/7bc2ad02-000000120.jpg
    2022-11-19 05:37:57.343 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 3
    2022-11-19 05:37:57.343 | INFO     | __main__:move_datasets_image_file:14 -            file_name : c374c1d1-000000116.jpg
    2022-11-19 05:37:57.344 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/c374c1d1-000000116.jpg
    2022-11-19 05:37:57.345 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/c374c1d1-000000116.jpg
    2022-11-19 05:37:57.384 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 4
    2022-11-19 05:37:57.385 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 1658a838-000000112.jpg
    2022-11-19 05:37:57.385 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/1658a838-000000112.jpg
    2022-11-19 05:37:57.386 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/1658a838-000000112.jpg
    2022-11-19 05:37:57.428 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 5
    2022-11-19 05:37:57.428 | INFO     | __main__:move_datasets_image_file:14 -            file_name : cc9038c8-000000108.jpg
    2022-11-19 05:37:57.429 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/cc9038c8-000000108.jpg
    2022-11-19 05:37:57.430 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/cc9038c8-000000108.jpg
    2022-11-19 05:37:57.468 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 6
    2022-11-19 05:37:57.471 | INFO     | __main__:move_datasets_image_file:14 -            file_name : c4def748-000000104.jpg
    2022-11-19 05:37:57.472 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/c4def748-000000104.jpg
    2022-11-19 05:37:57.472 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/c4def748-000000104.jpg
    2022-11-19 05:37:57.510 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 7
    2022-11-19 05:37:57.511 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 434f0155-000000100.jpg
    2022-11-19 05:37:57.512 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/434f0155-000000100.jpg
    2022-11-19 05:37:57.512 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/434f0155-000000100.jpg
    2022-11-19 05:37:57.552 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 8
    2022-11-19 05:37:57.553 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 3048621e-000000096.jpg
    2022-11-19 05:37:57.554 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/3048621e-000000096.jpg
    2022-11-19 05:37:57.555 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/3048621e-000000096.jpg
    2022-11-19 05:37:57.594 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 9
    2022-11-19 05:37:57.595 | INFO     | __main__:move_datasets_image_file:14 -            file_name : f4a6c0f3-000000092.jpg
    2022-11-19 05:37:57.595 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/f4a6c0f3-000000092.jpg
    2022-11-19 05:37:57.596 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/f4a6c0f3-000000092.jpg
    2022-11-19 05:37:57.631 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 10
    2022-11-19 05:37:57.633 | INFO     | __main__:move_datasets_image_file:14 -            file_name : d1d78546-000000088.jpg
    2022-11-19 05:37:57.633 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/d1d78546-000000088.jpg
    2022-11-19 05:37:57.634 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/d1d78546-000000088.jpg
    2022-11-19 05:37:57.675 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 11
    2022-11-19 05:37:57.676 | INFO     | __main__:move_datasets_image_file:14 -            file_name : d8a5a419-000000084.jpg
    2022-11-19 05:37:57.676 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/d8a5a419-000000084.jpg
    2022-11-19 05:37:57.677 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/d8a5a419-000000084.jpg
    2022-11-19 05:37:57.717 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 12
    2022-11-19 05:37:57.718 | INFO     | __main__:move_datasets_image_file:14 -            file_name : e3b48321-000000080.jpg
    2022-11-19 05:37:57.718 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/e3b48321-000000080.jpg
    2022-11-19 05:37:57.719 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/e3b48321-000000080.jpg
    2022-11-19 05:37:57.760 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 13
    2022-11-19 05:37:57.761 | INFO     | __main__:move_datasets_image_file:14 -            file_name : db0285a4-000000048.jpg
    2022-11-19 05:37:57.762 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/db0285a4-000000048.jpg
    2022-11-19 05:37:57.762 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/db0285a4-000000048.jpg
    2022-11-19 05:37:57.802 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 14
    2022-11-19 05:37:57.803 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 0a8d7fce-000000040.jpg
    2022-11-19 05:37:57.804 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/0a8d7fce-000000040.jpg
    2022-11-19 05:37:57.805 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/0a8d7fce-000000040.jpg
    2022-11-19 05:37:57.847 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 15
    2022-11-19 05:37:57.848 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 2535401d-000000036.jpg
    2022-11-19 05:37:57.848 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/2535401d-000000036.jpg
    2022-11-19 05:37:57.849 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/2535401d-000000036.jpg
    2022-11-19 05:37:57.891 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 16
    2022-11-19 05:37:57.892 | INFO     | __main__:move_datasets_image_file:14 -            file_name : ad9dd54e-000000016.jpg
    2022-11-19 05:37:57.893 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/ad9dd54e-000000016.jpg
    2022-11-19 05:37:57.894 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/ad9dd54e-000000016.jpg
    2022-11-19 05:37:57.932 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 17
    2022-11-19 05:37:57.933 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 44ca9f17-000000008.jpg
    2022-11-19 05:37:57.934 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/44ca9f17-000000008.jpg
    2022-11-19 05:37:57.935 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/44ca9f17-000000008.jpg
    2022-11-19 05:37:57.966 | INFO     | __main__:move_datasets_image_file:11 - >>>>>>>>>>>>                    i : 18
    2022-11-19 05:37:57.967 | INFO     | __main__:move_datasets_image_file:14 -            file_name : 0d4c392e-000000004.jpg
    2022-11-19 05:37:57.968 | INFO     | __main__:move_datasets_image_file:17 -          source_path : datasets/v0/images/0d4c392e-000000004.jpg
    2022-11-19 05:37:57.968 | INFO     | __main__:move_datasets_image_file:20 -          target_path : datasets/v0/val2017/0d4c392e-000000004.jpg



```python

```
