import cv2
import scipy.io as sio
import os
from centerface import CenterFace

# 顔認識器の設定
landmarks = True
centerface = CenterFace(landmarks=landmarks)

# 編集ファイルを開く
path_r = '/home/babaamata/workspace/Cat-faces-dataset/dataset-part3/'

# カウント変数
image_count = 0
face_count = 0

files = os.listdir(path_r)
for file in files:
  if (file.find('.png')!=-1):
    img = cv2.imread(path_r + file.replace('\n', '')) # 画像の読み込み

    # 顔認識
    h, w = img.shape[:2]
    if landmarks:
        dets, lms = centerface(img, h, w, threshold=0.35)
    else:
        dets = centerface(img, threshold=0.35)

    # 集計
    image_count = image_count + 1
    face_count = face_count + len(lms)

    # 図形の書き込み
    if len(lms) > 0:
      for det in dets:
          boxes, score = det[:4], det[4]
          cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
      if landmarks:
          for lm in lms:
              for i in range(0, 5):
                  cv2.circle(img, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
      cv2.imwrite("./output/part3/" + str(face_count)+ "_" + file, img)  # 画像の書き出し

print(image_count)
print(face_count)
