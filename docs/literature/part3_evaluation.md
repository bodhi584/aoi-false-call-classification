# Literature Part 3: Evaluation in Time Order, Changing Production Data, and Imbalanced Metrics

## 1. Gama et al. (2014) — Survey on Concept Drift Adaptation

* **Reference:** J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," *ACM Comput. Surv.*, vol. 46, no. 4, pp. 1–37, 2014, doi: 10.1145/2523813.
* **Summary:** This survey provides a comprehensive framework for learning in non-stationary environments where the statistical properties of the target variable change over time (concept drift). It categorizes adaptive learning mechanisms into drift detection, memory management, and model update strategies, while detailing evaluation methodologies such as prequential (interleaved test-then-train) analysis and time-ordered chunk evaluation.
* **Relates to our project:** Provides the theoretical foundation for using a day-by-day simulation (walk-forward evaluation) across the 132 recorded production days to handle time-varying component lots, machine wear, and optical measurement drift in the AOI dataset without data leakage.

## 2. He & Garcia (2009) — Learning from Imbalanced Data

* **Reference:** H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263–1284, Sep. 2009, doi: 10.1109/TKDE.2008.239.
* **Summary:** This paper systematically reviews learning algorithms and assessment metrics for highly skewed class distributions. It demonstrates why standard overall accuracy is misleading in imbalanced contexts and highlights metric trade-offs using recall (sensitivity), specificity, precision, and ROC/PR curves to evaluate minority class performance.
* **Relates to our project:** Guides our selection of recall and specificity as primary KPIs to evaluate the binary classification of imbalanced AOI alarms, where true defects constitute a rare positive class compared to majority false calls.

## 3. Lu et al. (2018) — Learning under Concept Drift: A Review

* **Reference:** J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang, "Learning under concept drift: A review," *IEEE Trans. Knowl. Data Eng.*, vol. 31, no. 12, pp. 2346–2363, Dec. 2019, doi: 10.1109/TKDE.2018.2876857.
* **Summary:** This review covers advancements in concept drift understanding, tracking, and adaptation. It formalizes drift taxonomy (sudden, gradual, incremental, reoccurring) and categorizes adaptation strategies into simple retraining windows, ensemble adjustments, and online parameter updates under time-ordered constraints.
* **Relates to our project:** Directly informs our design of historical retraining windows (e.g., rolling day windows) to adapt supervised machine learning models (logistic regression, random forest, single-hidden-layer neural network) to changing production distributions over time.

## 4. Fawcett (2006) — An Introduction to ROC Analysis

* **Reference:** T. Fawcett, "An introduction to ROC analysis," *Pattern Recognit. Lett.*, vol. 27, no. 8, pp. 861–874, Jun. 2006, doi: 10.1016/j.patrec.2005.10.010.
* **Summary:** This article explains Receiver Operating Characteristic (ROC) graph analysis, threshold tuning, and performance evaluation for binary decision systems. It details the explicit trade-off between True Positive Rate (recall) and False Positive Rate (1 - specificity) across varying decision thresholds.
* **Relates to our project:** Explains how operating decision thresholds can be tuned in supervised machine learning classifiers to guarantee high recall ($\ge 99\%$) for true defects while maximizing specificity ($\ge 20\%$) to filter out false calls.