

**Drift-Triggered Retraining (DDM)**
To manage concept drift in the continuous Siemens AOI data stream, the pipeline implements the Drift Detection Method (DDM) (Gama et al., 2004). DDM monitors the online error rate of the classification model by tracking the probability of error ($p_i$) and its standard deviation ($s_i$) as new instances are classified. The algorithm maintains the minimum values ($p_{min}$ and $s_{min}$) observed during execution. If the error rate increases significantly such that $p_i + s_i \ge p_{min} + 3s_{min}$, a concept drift is flagged. This statistical threshold ensures that natural variance in the AOI false-call stream is ignored, while a genuine shift in the defect distribution correctly triggers an immediate model retraining phase.



The Function: We wrote ddm.py, which calculates the running probability of errors and checks it against the baseline.

The Tests: We wrote three specific unit tests to ensure the function works under different conditions: one where the error rate is low and stable (no drift), one where the model suddenly fails (sudden drift), and one to ensure it waits for enough data before making a decision.

The Results: The output of your code is simply the index number of the exact moment the model starts failing (or -1 if it is still working fine).