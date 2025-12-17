# 桜の物体検出プロジェクト (Sakura Object Detection using YOLO)

## 1. プロジェクト紹介 (Introduction)
本プロジェクトは、「ディープラーニング——TensorFlowに基づく人工知能の実践応用」コースの期末課題として実施されたものです 。
物体検出技術は幅広いシーンで応用されており、本研究では品種が多く汎用性が高い「桜」を対象としました 。YOLOシリーズのモデル（v8, v9, v11）を用いて桜の検出を行い、アルゴリズムの原理理解と実装、そして精度の比較を行いました 。

## 2. ファイル構成 (File Descriptions)
プロジェクトの主なファイル構成は以下の通りです：

* **Dataset/**: 学習およびテスト用の画像データ（YOLO形式）。
    * [cite_start]`train/`: 学習用データセット [cite: 23, 24]
    * [cite_start]`test/`: テスト用データセット [cite: 25, 26]
    * [cite_start]`valid/`: 検証用データセット [cite: 27, 28]
* [cite_start]**data.yaml**: データセットのパス、クラス数、ラベル名を定義した設定ファイル [cite: 29]。
* **Notebooks/**:
    * [cite_start]モデルのトレーニングおよび評価を実行する Jupyter Notebook ファイル [cite: 16]。
* [cite_start]**runs/**: 学習過程のログ、混同行列、F1曲線、PR曲線などの結果画像が保存されるディレクトリ [cite: 93-104]。

## 3. データセット (Dataset)
* **データソース**: Roboflow (Sakura-jklcu) 
* **データ量**: 1400枚以上のサンプルが含まれています 。
* **形式**: YOLOフォーマット（位置とラベルがタグ付け済み） 。
* **前処理**:
    * 解凍後のファイル名が文字化けしていたため、`00001.jpg`, `00001.txt` のような連番形式にリネーム処理を実施しました 。
    * データセットは既にYOLOモデルに適応しているため、その他の特別な処理は行っていません 。

## 4. 使用した手法・モデル (Methods)
本プロジェクトでは、YOLO（You Only Look Once）モデルの異なるバージョンを比較・検討しました。

### **YOLOv8** 
* **特徴**: アンカーフリー、マルチスケール特徴融合。
* **構造**: Backboneに `C2f` モジュールを採用し、軽量化と勾配情報の豊かなフローを実現。SPPFによる異なるスケールのプーリングを使用。

### **YOLOv9** 
* **特徴**: YOLOv8の構造をベースに改良。
* **構造**: `C2f` を `RepNCSPELAN4` に変更。新しい分岐（Auxiliary branch）を追加し、バックボーンの学習を促進する仕組み（CBFuseなど）を導入 。

### **YOLOv11** 
* **特徴**: 最新の改良版（本プロジェクトで最良の結果）。
* **構造**:
    * `C3k2`: マルチスケール畳み込み核を導入し、広範なコンテキスト情報を捕捉 。
    * `C2PSA`: SPPFの後に配置された自己注意（Self-Attention）モジュール。PSAメカニズムによりマルチスケール特徴を抽出・融合 。

## 5. 実装手順 (Implementation)

### 開発環境
* Python 3.9 
* Anaconda / Jupyter Notebook 
* PyTorch 

### トレーニング設定
モデルのトレーニングは以下の設定で行いました ：

```python
model_yolov8.train(data='data.yaml', epochs=50, batch=1, task='detect+classify')
