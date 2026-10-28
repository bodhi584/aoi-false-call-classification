2.2 Classifiers for False Calls
2.2.1 Enhancing False Call Detection in Automated Optical Inspection Using XGBoost Classifier and XAI Algorithms

Authors: J. Albayari et al. (2025), Flexible Automation and Intelligent Manufacturing (FAIM) [1]
This paper addresses the high rate of false calls in automated optical inspection (AOI) systems by applying an XGBoost classifier to surface mount technology production data. The authors utilize explainable AI algorithms to interpret the predictions of the model and understand feature contributions.Key Insight: Gradient boosting models can effectively filter production data to identify false calls while providing necessary interpretability for manufacturing environments.Relevance: Our project builds on this by performing binary classification of AOI alarms into true defects and false calls using the exact same Siemens dataset.

2.2.2 Scikit-learn: Machine Learning in Python

Authors: F. Pedregosa et al. (2011), Journal of Machine Learning Research [2]
This article introduces scikit-learn, a highly optimized Python library that integrates a wide range of state-of-the-art machine learning algorithms. The framework is designed to provide an accessible, consistent, and efficient interface suited for medium-scale data analysis and traditional modeling tasks.Key Insight: Scikit-learn provides a robust, standardized environment for implementing, tuning, and comparing various machine learning algorithms.Relevance: We rely entirely on scikit-learn to train and evaluate our supervised machine learning baseline models, including logistic regression, random forest, and a neural network with one hidden layer.2.2.3 A Machine Learning Based Approach to Detect False Calls in SMT Manufacturing

Authors: N. Thielen et al. (2020), 43rd International Spring Seminar on Electronics Technology (ISSE) [3]
This study evaluates multiple machine learning classifiers using component-level measurements from AOI machines to address highly imbalanced datasets in manufacturing. The authors compare neural networks, k-nearest neighbors, and random forests, concluding that random forests achieve the best overall performance for defect detection.Key Insight: Random forests and neural networks are highly capable of managing the severe class imbalance inherent in AOI inspection data.Relevance: This directly supports our methodology of testing random forests and neural networks for the binary classification of true defects versus false calls.

2.3 Addressed Evaluation Gap
Current literature demonstrates that supervised machine learning can successfully filter AOI false calls. However, models are typically evaluated using randomized train-test splits that ignore actual production timelines and data drift. Our project addresses this critical gap by implementing a strict day-by-day simulation (walk-forward evaluation) to test models in chronological order. Furthermore, to prioritize production safety, we evaluate our models' specificity exclusively under the strict condition of maintaining a fixed minimum recall of 99% for true defects.
