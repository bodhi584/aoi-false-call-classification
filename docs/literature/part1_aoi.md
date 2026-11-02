# Literature part 1: AOI and false calls in electronics assembly

Team 1 · issue #31 · author: Melisa Subasi

AOI (automated optical inspection) checks every solder joint on a circuit board with a camera. It compares the measurements with fixed limits. Boards that are flagged go to a manual inspection station, and most of these alarms are false calls. The four sources below describe our data, the interest of industry today, and two earlier studies on the same problem.

## [1] Pfab et al. (2024): our data set

K. Pfab, R. Eichler, A. Mallandur, and M. Rothering, "Data of automated optical inspection of surface-mounted technology electronic production," *Data in Brief*, vol. 53, Art. no. 110110, Apr. 2024, doi:10.1016/j.dib.2024.110110.

Authors from Siemens and the University of Helsinki share 132 days of data from one real production line: 440,274 flagged solder joints with about 70 measurements each, labelled true defect or false call. Only 0.3 % to 2.8 % of the rows are true defects, depending on the inspection type. The authors warn that some labels may be wrong (one operator set each label) and that the data may change over time.

*Relation to our project:* this is our data set. Wrong labels and changes over time are the reasons why we test in time order and why 99 % recall may be hard to keep.

## [2] Siemens and NVIDIA (21 Oct 2026): a problem of today

Siemens, "Siemens and NVIDIA advance AI-powered PCB inspection from Digital Twin to factory," Press release, Plano, TX, USA, Oct. 21, 2026. [Online]. Available: https://news.siemens.com/en-us/siemens-nvidia-pcba-digital-twin-factory/

Siemens and NVIDIA work together to prepare the optical inspection of circuit boards on the computer, before the first real board is made; this is shown at the Siemens factory in Erlangen. When the AOI finds a possible defect, an AI check helps to decide if it is a real defect or a false call. The project targets 85 % less manual engineering work, 40 % fewer false calls and more than 5 % better first-pass yield.

*Relation to our project:* our project makes the same decision (real defect or false call) with public data. The 40 % is a target, not a result.

## [3] Thielen et al. (2020): machine learning on AOI measurements

N. Thielen, D. Werner, K. Schmidt, R. Seidel, A. Reinhardt, and J. Franke, "A machine learning based approach to detect false calls in SMT manufacturing," in *Proc. 43rd Int. Spring Seminar on Electronics Technology (ISSE)*, Demanovska Valley, Slovakia, May 2020, pp. 1–6, doi:10.1109/ISSE49702.2020.9121044.

The authors test if a machine learning step after the AOI can find false calls. They use AOI measurements (offset, angle and score of each part) and compare k-nearest neighbours, a neural network and a random forest on a small data set (1,144 parts) and a larger one (4,264 parts). The random forest has the best accuracy (0.92 and 0.94), but it cannot reach zero missed defects with a threshold, while the neural network can; with a high threshold, about 25 % of false calls are found with no missed defect.

*Relation to our project:* it is the same task on the same kind of data. It also sets the number of missed defects first and counts the saved checks second, like our 99 % recall rule. Its data set is small, ours is much larger.

## [4] Pfab and Rothering (2025): testing on the same data

K. Pfab and M. Rothering, "Towards improved research methodologies for industrial AI: A case study of false call reduction," arXiv:2506.14521 [cs.LG], Jun. 17, 2025. [Online]. Available: https://arxiv.org/abs/2506.14521

The authors use false call reduction in AOI to show seven common weaknesses in industrial AI research, for example no shared data or code, wrong metrics, no clear success goal and no check of change over time. They train several models (for example k-nearest neighbours, random forest and XGBoost) on the Siemens data from [1], with the goal of at least 40 % fewer manual checks and at most 1 % missed defects. Two models reach the goal on a normal test set, but on later time slices the missed defects rise to about 5 % to 15 % and the models fail.

*Relation to our project:* it uses the same data as we do and asks for tests in time order and for clear goals. This supports our plan: we fix the recall first and test day by day in time order. It also shows that our 99 % recall goal can be hard to keep on later days.
