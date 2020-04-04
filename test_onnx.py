# coding:utf-8
import cv2
import numpy as np
import onnxruntime

# cocoデータセットのクラス番号：クラス名
coco_classes = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41:'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush'
}

# モデルを読み込む
session = onnxruntime.InferenceSession("./ssdlite_mobilenetv2.onnx")

# 画像を読み込む
img = cv2.imread("./pic/demo.png")

# OpenCVは画像データをBGRで読み込むので、BGRに変換する
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

width, height = img.shape[0], img.shape[1]

img_data = np.expand_dims(img, axis=0)

# モデルの推論の準備
input_name = session.get_inputs()[0].name   # 'image'
output_name_boxes = session.get_outputs()[0].name     # 'boxes'
output_name_classes = session.get_outputs()[1].name   # 'classes'
output_name_scores = session.get_outputs()[2].name    # 'scores'
output_name_num = session.get_outputs()[3].name       # 'number of detections'

# 推論
outputs_index = session.run(
    [output_name_num, output_name_boxes, output_name_scores, output_name_classes],
    {input_name: img_data}
)

# 結果を受け取る
output_num = outputs_index[0] # 検出した物体数
output_boxes = outputs_index[1] # 検出した物体の場所を示すボックス
output_scores = outputs_index[2] # 検出した物体の予測確率
output_classes = outputs_index[3] # 検出した物体のクラス番号

# 予測確率の閾値
threshold = 0.6

# 推論結果を画像に変換
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for detection in range(0, int(output_num[0])):
    if output_scores[0][detection] > threshold:
        classes = output_classes[0][detection]
        boxes = output_boxes[0][detection]
        scores = output_scores[0][detection]
        top = boxes[0] * width
        left = boxes[1] * height
        bottom = boxes[2] * width
        right = boxes[3] * height

        top = max(0, top)
        left = max(0, left)
        bottom = min(width, bottom)
        right = min(height, right)
        img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0,0,255), 3)
        img = cv2.putText(
        	img, "{}: {:.2f}".format(coco_classes[classes], scores),
        	(int(left), int(top)),
        	cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
        )

cv2.imwrite("./pic/result.png", img)
