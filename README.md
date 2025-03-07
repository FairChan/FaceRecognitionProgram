# FaceRecognitionProgram
Private Program for XMUM class G0191 Face Recognition with Machine Learning.

******************************************* 
* 提供中英双语readme文件			  *
* Provide English and Chinese readme file *
*******************************************

There are two models for this project:
The first model, the original model: use the code to rotate, flip, blur, and noise the image to generate more different images as training data. Haar features are extracted using opencv and adaboast, dimensionality is reduced using pca, and then a classifier is used which requires us to feed face images and non-face images to train out the model, the images fed include: non-face, unnamed face, named face. As the model is outdated, it is difficult to recognize named faces, however, increasing the accuracy (retaining more variance information, increasing the uniform face image size) requires exponentially more power (feeding 10,000 images takes at least one hour on a 16G unified memory 10-core M2 chip), so we switch to a more advanced model to ensure higher accuracy and less power for recognition.
The second model, the advanced model: we use images that have been processed in advance as the training set, based on DeepFace (using ArcFace to extract 512-dimensional face feature vectors) and MTCNN for face detection, and at the same time, we utilize OpenCV for image preprocessing, to ensure that the input image size is the same. Identity labels are extracted by parsing file names and unknown faces are classified as “unknown” to achieve data categorization. PCA dimensionality reduction (50 dimensions) is used to optimize computational efficiency, and SVM (linear kernel) is used as a classifier for face recognition, while probabilistic prediction is supported. The system supports incremental training, which automatically reads the existing feature data (if it exists), merges the old and new data, and retrains the SVM to avoid full retraining and improve training efficiency. Finally, the model and data (PCA model, SVM classifier, face features, label encoder) are stored by pickle, which is easy to load and update quickly.

Operational overview:
In the first model, the dataset folder holds the training data, containing named (named faces), unkown (unnamed faces), and negative (non-faces). The various testing folders store test images with different levels of blur. face_recognition_model.pkl is the stored model. The model is trained by feeding different images through train.py, and the various images in testimg are recognized through predict.py. For fast testing, all the image files under TEST_PATH in predict.py are tested and given a True Positive Rate.
In the second model, there are different levels of blurred images in the input folder, these files are used for training and embeddings_data.pkl,face_classifier.pkl,label_encoder.pkl are generated after training. About the test image: running Predict.py will prompt for the path to the test image.

这个项目有两个模型：
第一个模型，原始模型：使用代码将图像进行旋转，翻转，模糊，噪声处理后，生成更多不同的图片作为训练数据。用opencv和adaboast提取Haar特征，使用pca降低维度，然后使用分类器需要我们喂人脸图和非人脸图片训练出模型，投喂的图片包括：非人脸，未命名人脸，已命名人脸。由于模型较为落后，难以识别已命名人脸，然而提高精度（保留更多方差信息,提高统一的人脸图片大小）需要消耗指数增长的算力（投喂10，000张图片在16G统一内存10核的M2芯片上需要运行至少一小时），因此我们换用更为先进的模型以保证识别的较高的精度和较少的算力。
第二个模型，先进模型：用已经提前处理好的图片作为训练集，基于 DeepFace（使用 ArcFace 提取 512 维人脸特征向量）和 MTCNN 进行人脸检测，同时利用 OpenCV 进行图像预处理，确保输入图像尺寸一致。通过解析文件名提取身份标签，并将未知人脸分类为 "unknown"，实现数据归类。采用 PCA 降维（50 维） 以优化计算效率，并使用 SVM（线性核） 作为分类器进行人脸识别，同时支持概率预测。系统支持 增量训练，会自动读取已有的特征数据（若存在），合并新旧数据后重新训练 SVM，避免全量重训，提高训练效率。最终，模型及数据（PCA 模型、SVM 分类器、人脸特征、标签编码器）均通过 pickle 存储，便于后续快速加载与更新。

操作概述：
第一个模型中，dataset文件夹存放的是训练数据，包含named（已命名的人脸），unkown（未命名的人脸），negative（非人脸）。其中各种testing文件夹下存放的是不同模糊程度的测试图片。face_recognition_model.pkl是存储的模型。通过train.py来投喂不同的图片训练模型，通过predict.py来识别testimg中的各种图片，为实现快速测试，predict.py中TEST_PATH下的所有图片文件都会进行测试，并给出True Positive Rate
第二个模型中，input文件夹中有不同程度的模糊图片，此些文件用于训练，而embeddings_data.pkl,face_classifier.pkl,label_encoder.pkl都是训练后生成的模型。关于测试图片：运行Predict.py会提示输入测试图片路径。