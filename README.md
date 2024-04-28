# Supervised ML and DL methods for anomaly detection in IOT to enahnce network security
This project aims to evaluate the effectiveness of supervised machine-learning-based and deep-learning-based anomaly detection techniques in distinguishing between benign and malicious network traffic in the IoT-23 dataset and to assess the robustness and scalability of different supervised anomaly detection methods in handling the dynamic and heterogeneous nature of IoT network environments.


## Data Set (Aposemat IoT-23)
The dataset used in this project is: [iot_23_datasets_small](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz).<br/>
- The lighter version containing only the labeled flows without the pcaps files (8.8 GB) [Aposemat IoT-23 dataset](https://www.stratosphereips.org/datasets-iot23).
- A labeled dataset with malicious and benign IoT network traffic.
- This dataset was created as part of the Avast AIC laboratory with the funding of Avast Software. 

## Data Classification Details
The project is implemented in four distinct steps simulating the essential data processing and analysis phases. <br/>
- Each step is represented in a corresponding Jupyter Notebook inside [IOT-23-dataset-extraction](IOT-23-dataset-extraction).
- Data files (raw, interim, processed) are stored inside the [CSV-data](CSV-data) path.
- Trained data models are stored inside [applied-ML-DL-methods](applied-ML-DL-methods).

### PHASE 1 - Data Cleaning and Processing
> Corresponding Jupyter Notebook:  [iot-23-data-cleaning-and-preprocessing.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-cleaning-and-preprocessing.ipynb)

Implemented data exploration and cleaning tasks:
1. Loading the raw dataset file into pandas DataFrame.
2. Exploring dataset summary and statistics.
3. Fixing combined columns.
4. Dropping irrelevant columns.
5. Fixing unset values and validating data types.
6. Checking the cleaned version of the dataset.
7. Storing the cleaned dataset to a csv file.
8. Exploring dataset summary and statistics.
9. Analyzing the target attribute.
10. Encoding the target attribute using [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
11. Encoding IP addresses.
12. Handling missing values:
    1. Impute missing categorical features using [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
    2. Impute missing numerical features using [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).
13. Encoding categorical features: handling rare values and applying [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
14. Scaling numerical attributes using [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
15. Checking the processed dataset and storing it to a csv file.

### PHASE 2 - Data Training
> Corresponding Jupyter Notebook:  [iot-23-data-training.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-training.ipynb)

Trained and analyzed classification models:
1. Naive Bayes: [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
2. K-Nearest Neighbors: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
3. Decision Tree: [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
4. Random Forest: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
5. LinearSVC: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
6. Artificial Neural Network (ANN): [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
7. AdaBoost: [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
8. XGBoost: [XGBClassifier](https://xgboost.readthedocs.io/en/stable/index.html#)

Evaluation method: 
- Cross-Validation Technique: [Stratified K-Folds Cross-Validator](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- Folds number: 5
- Shuffled: Enabled

Results were analyzed and compared for each considered model.<br/>

### PHASE 3 - Data Tuning
> Corresponding Jupyter Notebook:  [iot-23-data-tuning-Naive-Bayes.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-Naive-Bayes.ipynb)

Model tuning details:
- Tuned model: Naive Bayes - [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-KNN.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-KNN.ipynb)

Model tuning details:
- Tuned model: KNN - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-Decision-Tree.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-Decision-Tree.ipynb)

Model tuning details:
- Tuned model: Decision Tree - [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-Random-Forest.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-Random-Forest.ipynb)

Model tuning details:
- Tuned model: Random Forest - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-SVC.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-SVC.ipynb)

Model tuning details:
- Tuned model: LinearSVC - [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-ANN.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-ANN.ipynb)

Model tuning details:
- Tuned model: ANN - [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-AdaBoost.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-AdaBoost.ipynb)

Model tuning details:
- Tuned model: AdaBoost - [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.

> Corresponding Jupyter Notebook:  [iot-23-data-tuning-XGBoost.ipynb](https://github.com/AmazingCoder107856/Machine-and-Deep-Learning-methods-anomaly-detection-using-IoT-23-dataset/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning-XGBoost.ipynb)

Model tuning details:
- Tuned model: XGBoost - [XGBClassifier](https://xgboost.readthedocs.io/en/stable/index.html#)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.