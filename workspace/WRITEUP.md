# Project Write-Up

## Explaining Custom Layers

According to some researches, I pick faster_rcnn_inception_v2_coco_2018_01_28 over other models in the Tennsorflow Model Zoo here(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  and the OpenCV zoo here(https://docs.openvinotoolkit.org/2019_R1/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) ].
Overal, the faster_rcnn_inception_v2_coco_2018_01_28 perform better in accuaracy. Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.

Steps: 
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
```
cd faster_rcnn_inception_v2_coco_2018_01_28
```
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
## Comparing Model Performance
My methods to compare models before and after conversion to Intermediate Representations were based on size of model, max-min accuracy, inference time, network need, cost of using cloud service. I do average all the probability every time the Inference detect a person. I also put a timer before it takes the frame to the Inference and after it resturn the result. I implemented using() to calculate the memory that it currently using.

The difference between model accuracy pre- and post-conversion was similiar. Both of the models yield over 90% probability on average. 

The memory usage is significantly different. The post-conversion model used 210 MB(before: 56MB After: 266). The pre-trained model used 72 MB(before: 54MB after 126MB)

The size of the model pre- and post-conversion was significantly different. For the faster_rcnn_inception_v2_coco_2018_01_28, pre-conversion is 57 MB, post-conversion is 51 MB and for the pre-trained person-detection-retail-0013 is nearly 2 MB.

The inference time of the model pre- and post-conversion was significantly different. For the faster_rcnn_inception_v2_coco_2018_01_28, a pure coco tf, it takes average time 960ms to process. Meanwhile, the pre-trained model person-detection-retail-0013 takes average 46ms.

For the network, notice the output above, we can see that the faster_rcnn_inception_v2_coco_2018_01_28 barely works with the server since it's slow to load the next frame. Mean while the person-detection-retail-0013 work better.

If we have to deploy those on a cloud server, we need to consider more delay time and data loss. Thus, we will experience lagging and loss frame with faster_rcnn_inception_v2_coco_2018_01_28. We will experience less loss with the other model.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are used for smart store like new Amazon Grocery store. Or any detection need to count the people number during the panndemic COVID-19.

Each of these use cases would be useful because we decentralizing suppervision anywhere, even that places don't have internet.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:
+ This might not work at night, evening, rain, snow.
+ Human can camourflage, thus, low accuracy
