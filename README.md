# PCB-Defect-Detection
此專案為使用YOLOv5偵測PCB板上的缺陷種類  
類別種類有missing hole, mouse bite, open circuit, short, spur, spurious copper

## 資料集
因為dataset的資料量比較少，所以透過資料增強(data augmentation)增加訓練用的數據，在資料增強的過程中，bbox會根據圖片的旋轉或縮放而自動調整，因此我們不需要額外花時間標記ground truth。
```
python aug_with_bbox.py
```

## 結果
<img src="img/01_missing_hole_01.jpg" width="100%"/>
<img src="img/01_open_circuit_02.jpg" width="100%"/>
<img src="img/01_short_03.jpg" width="100%"/>
