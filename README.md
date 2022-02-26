## 一、项目背景
![](https://ai-studio-static-online.cdn.bcebos.com/bc42c437ccb84145b99b3a1fa0ce70929fa16a9d417f47d282d948d7e5bb851d)


* 如图所示，在一些化工实验室，许多化工原料都标上了特定的编号，这样做虽然易于管理，但想要知道原料的信息往往需要根据编号去查询相关的资料，这样多少存在一些不便，特别在实验过程中如果忘记了原料的一些注意事项或剂量，就不得不停止实验去进行查询，这样会对实验进行较大的影响，再者，现实生产过程中，所谓的原料往往成分并不单一，它是由专门的原料生产商进行合成并出售，以供相关的化工企业进行采购和二次生产，这些原料大多没有确定的名称(合成品的缘故)编号也不统一（取决于生成商,不同生产商或许会有不同的编码制度),由于成分以及编码的特殊性，找到一种便捷的原料信息查询方式非常重要，因此本项目打算结合ai，基于目标检测+OCR实现原料信息的快速查询,免去人工负担，提高生产效率。



## 二、数据处理及相关配置


```python
#解压数据集
!unzip -oq /home/aistudio/data/data128635/bottle.zip
```


```python
#下载paddledetection
!git clone https://gitee.com/paddlepaddle/PaddleDetection.git
```

    Cloning into 'PaddleDetection'...
    remote: Enumerating objects: 21396, done.[K
    remote: Counting objects: 100% (1866/1866), done.[K
    remote: Compressing objects: 100% (934/934), done.[K
    remote: Total 21396 (delta 1312), reused 1302 (delta 929), pack-reused 19530[K
    Receiving objects: 100% (21396/21396), 202.22 MiB | 8.56 MiB/s, done.
    Resolving deltas: 100% (15861/15861), done.
    Checking connectivity... done.



```python
# 配置
!pip install paddlepaddle-gpu
!pip install pycocotools
!pip install lap
!pip install motmetrics
import cv2
from matplotlib import pyplot as plt
import numpy
```


```python
#数据集划分
import random
import os
#生成train.txt和val.txt
random.seed(2020)
xml_dir  = '/home/aistudio/bottle/Annotations'#标签文件地址
img_dir = '/home/aistudio/bottle/images'#图像文件地址
path_list = list()
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img)
    xml_path = os.path.join(xml_dir,img.replace('jpg', 'xml'))
    path_list.append((img_path, xml_path))
random.shuffle(path_list)
ratio = 0.9
train_f = open('/home/aistudio/bottle/train.txt','w') #生成训练文件
val_f = open('/home/aistudio/bottle/val.txt' ,'w')#生成验证文件

for i ,content in enumerate(path_list):
    img, xml = content
    text = img + ' ' + xml + '\n'
    if i < len(path_list) * ratio:
        train_f.write(text)
    else:
        val_f.write(text)
train_f.close()
val_f.close()

#生成标签文档
label = ['bottle']#设置你想检测的类别
with open('/home/aistudio/bottle/label_list.txt', 'w') as f:
    for text in label:
        f.write(text+'\n')
```

![](https://ai-studio-static-online.cdn.bcebos.com/4843bae2c0ea4d9ea4bde4473d06a4194e9b193fe2ef41038853904dbd56c7f5)



## 三、基于PaddleDetection实现检测

* 本项目基于骨干网络为yolov3_darknet53_270e_voc的yolov3模型进行训练，训练前的相关配置，请按需调整，在此不做赘述。


```python
%cd 
```

    /home/aistudio



```python
#模型训练
!python /home/aistudio/PaddleDetection/tools/train.py -c /home/aistudio/PaddleDetection/configs/yolov3/yolov3_darknet53_270e_voc.yml --eval --use_vdl=True --vdl_log_dir="./output"

```

![](https://ai-studio-static-online.cdn.bcebos.com/bb4b9a0c37da4acda94d2c73c597d08e8157e2213b9f42aba3f4876d0b87d5e2)




```python
%cd PaddleDetection

```

    /home/aistudio/PaddleDetection



```python
#模型导出
!python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml  -o weights=../output/yolov3_darknet53_270e_voc/best_model.pdparams

```

    Warning: import ppdet from source directory without installing, run 'python setup.py install' to install ppdet firstly
    [02/26 20:44:40] ppdet.utils.checkpoint INFO: Finish loading model weights: ../output/yolov3_darknet53_270e_voc/best_model.pdparams
    [02/26 20:44:40] ppdet.engine INFO: Export inference config file to output_inference/yolov3_darknet53_270e_voc/infer_cfg.yml
    W0226 20:44:42.895236  3477 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0226 20:44:42.895300  3477 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [02/26 20:44:48] ppdet.engine INFO: Export model and saved in output_inference/yolov3_darknet53_270e_voc



```python
#模型预测
result=!python ./deploy/python/infer.py --model_dir=./output_inference/yolov3_darknet53_270e_voc --image_file=../1024.jpg 
# 输出结果
print(result)
```

    ['-----------  Running Arguments -----------', 'batch_size: 1', 'camera_id: -1', 'cpu_threads: 1', 'device: cpu', 'enable_mkldnn: False', 'image_dir: None', 'image_file: ../1024.jpg', 'model_dir: ./output_inference/yolov3_darknet53_270e_voc', 'output_dir: output', 'reid_batch_size: 50', 'reid_model_dir: None', 'run_benchmark: False', 'run_mode: paddle', 'save_images: False', 'save_mot_txt_per_img: False', 'save_mot_txts: False', 'scaled: False', 'threshold: 0.5', 'trt_calib_mode: False', 'trt_max_shape: 1280', 'trt_min_shape: 1', 'trt_opt_shape: 640', 'use_dark: True', 'use_gpu: False', 'video_file: None', '------------------------------------------', '-----------  Model Configuration -----------', 'Model Arch: YOLO', 'Transform Order: ', '--transform op: Resize', '--transform op: NormalizeImage', '--transform op: Permute', '--------------------------------------------', 'class_id:0, confidence:0.7390, left_top:[46.08,310.79],right_bottom:[407.12,952.04]', 'save result to: output/1024.jpg', '------------------ Inference Time Info ----------------------', 'total_time(ms): 9788.0, img_num: 1', 'average latency time(ms): 9788.00, QPS: 0.102166', 'preprocess_time(ms): 95.40, inference_time(ms): 9692.50, postprocess_time(ms): 0.10']


![](https://ai-studio-static-online.cdn.bcebos.com/7d89d5b76cc54b57ad62e90f4029caa95f78a466392f4154800293fe5b267835)


# PS：
# 本项目先对原料瓶进行了目标检测，这样做有什么好处呢？ 
# 答：OCR文字识别会对所有的文字信息进行提取，在实际生活中，具有文字信息的东西有很多，如果直接运用OCR进行文字识别，很有可能会识别到其他无用的文字信息，这是不利于我们的任务的，因而我们在OCR识别前增加了对原料瓶的目标检测步骤，意图是使机器最终提取的是我们需要的原料编号信息。

## 四、切分图像


```python
# 提取识别图像的坐标方便切分
import numpy as np
index1='class_id'
index2='right_bottom'
for dt in result:
      if index1 in dt:
          temp=dt
          break
print(temp)

b = temp.split(',')
x1=int(float(b[2][11:]))
y1=int(float(b[3][:-1]))
x2=int(float(b[4][14:]))
y2=int(float(b[5][:-1]))


print(x1,y1)
print(x2,y2)
```

    class_id:0, confidence:0.7390, left_top:[46.08,310.79],right_bottom:[407.12,952.04]
    46 310
    407 952



```python
# 读取图像
img =np.array(cv2.imread('/home/aistudio/1024.jpg'))
print(img.shape)
# 切分时适当扩大范围保证图像完整
img2 = img[y1-20:y2+20,x1-20:x2+20]
plt.imshow(img2)


```

    (1280, 960, 3)





    <matplotlib.image.AxesImage at 0x7efc0c7ed250>




![png](output_19_2.png)


## 五、使用paddlehub的ocr模型对切分图片进行快速推理


```python
#安装paddlehub以及相关库
!pip install --upgrade paddlepaddle -i https://mirror.baidu.com/pypi/simple
!pip install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple
!pip install shapely -i https://pypi.tuna.tsinghua.edu.cn/simple 
!pip install pyclipper -i https://pypi.tuna.tsinghua.edu.cn/simple 
```


```python
#导入相关库
import paddlehub as hub
import cv2
import numpy as np
import  matplotlib.pyplot as plt # plt 用于显示图片
import  matplotlib.image as mpimg # mpimg 用于读取图片
import  numpy as np

```


```python
ocr = hub.Module(name="chinese_ocr_db_crnn_server") 
# 加载模型
```

    Download https://bj.bcebos.com/paddlehub/paddlehub_dev/chinese_ocr_db_crnn_server-1.1.2.tar.gz
    [##################################################] 100.00%
    Decompress /home/aistudio/.paddlehub/tmp/tmppfsvzv99/chinese_ocr_db_crnn_server-1.1.2.tar.gz
    [##################################################] 100.00%


    [2022-02-20 09:38:22,941] [    INFO] - Successfully installed chinese_ocr_db_crnn_server-1.1.2
    [2022-02-20 09:38:23,007] [ WARNING] - The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object
    W0220 09:38:23.017434   101 analysis_predictor.cc:1350] Deprecated. Please use CreatePredictor instead.



```python
# 将切分图像送入ocr模型
np_images =[img2]  
# 使用paddlehub的ocr模型进行预测推理
results = ocr.recognize_text(
                    images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                    visualization=True,       # 是否将识别结果保存为图片文件；
                    box_thresh=0.5,           # 检测文本框置信度的阈值；
                    text_thresh=0.5)          # 识别中文文本置信度的阈值；

for result in results:
    data = result['data']
    save_path = result['save_path']
    for infomation in data:
        print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ', infomation['text_box_position'])
        if  '物料编号' in infomation['text']:
            code = infomation['text']
            print(code)
            break
# 获取编号索引
```

    text:  物料名称 
    confidence:  0.9979156255722046 
    text_box_position:  [[125, 287], [208, 284], [209, 308], [127, 311]]
    text:  物料编号：A0001 
    confidence:  0.9943730235099792 
    text_box_position:  [[128, 342], [254, 327], [257, 351], [132, 366]]
    物料编号：A0001



```python
# 获取编号索引
code2 = code[-5:]
print(code2)
```

    A0001



```python
#根据编号查询数据库
f=open('/home/aistudio/{}.txt'.format(code2), encoding='utf-8')
for line in f:
    print(line)

```

    编号:A0001(I1008)
    
    组分：金黄洋甘菊(CHRYSANTHELLUM INDICUM)提取物             0.1
    
              水                                           0.88
    
              1,2-己二醇                                    0.01
    
              对经基苯乙酮                                   0.01
    
    
    
    用途：皮肤调理剂
    
    备注：经识别不含安全性风险物质


## 六、总结与改进
1. 本项目基于检测和ocr实现了对原料信息的快速查询，但目前模型的精度仍然有待提高，数据集准备仍不够充足，需要进一步的完善，后续将扩充原料瓶类别，以保证更好的满足实际需求。
2. 除了查询功能外，本项目也可基于对原料的识别，对实验过程进行一个实时的记录与指导（可设置展示实验的合成路线，对实验的用料进行识别，当用户在实验过程中出现使用错误原料的情况，机器可实时提醒，当用户忘记原料信息时，机器也可实时查询。）


## 七、个人简介
> 作者：萧泽锋


> 嘉应学院 2018级 本科生


> 感兴趣方向：计算机视觉、深度学习


> 我在[AI Studio](https://aistudio.baidu.com/aistudio/usercenter)上获得白银等级，来互关呀~ 
