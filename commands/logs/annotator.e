2025-03-11 19:49:42,768 - pyexeggutor - INFO - Loading MNS
2025-03-11 19:50:12,058 - pyexeggutor - INFO - Annotating embeddings
Running bayesian AutoML to identify relevant features:   0%|          | 0/47 [00:00<?, ?it/s]Running bayesian AutoML to identify relevant features:   0%|          | 0/47 [01:08<?, ?it/s]
Traceback (most recent call last):
  File "/home/ec2-user/SageMaker/projects/mns-dev/niche-space/commands/run_annotator.py", line 55, in <module>
    annotator.fit(
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/nichespace/manifold.py", line 1736, in fit
    model_automl = self._run_regression_automl(
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/nichespace/manifold.py", line 1664, in _run_regression_automl
    model.fit(
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/clairvoyance/bayesian.py", line 585, in fit
    self._fit(
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/clairvoyance/bayesian.py", line 442, in _fit
    model_dcf.fit(X.iloc[indices_training])
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/feature_engine/selection/drop_constant_features.py", line 192, in fit
    self.features_to_drop_ = [
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/feature_engine/selection/drop_constant_features.py", line 193, in <listcomp>
    feature for feature in self.variables_ if X[feature].nunique() == 1
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/pandas/core/base.py", line 1065, in nunique
    uniqs = remove_na_arraylike(uniqs)
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/pandas/core/dtypes/missing.py", line 723, in remove_na_arraylike
    return arr[notna(arr)]
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/pandas/core/arrays/sparse/array.py", line 1015, in __getitem__
    return self.take(np.arange(n)[mask])
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/pandas/core/arrays/sparse/array.py", line 1055, in take
    return self._take_without_fill(indices)
  File "/home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/pandas/core/arrays/sparse/array.py", line 1132, in _take_without_fill
    if (indices.max() >= n) or (indices.min() < -n):
KeyboardInterrupt
Command terminated by signal 2
	Command being timed: "python run_annotator.py"
	User time (seconds): 88.29
	System time (seconds): 16.04
	Percent of CPU this job got: 101%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 1:42.43
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 28432296
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 31
	Minor (reclaiming a frame) page faults: 5249196
	Voluntary context switches: 95
	Involuntary context switches: 219
	Swaps: 0
	File system inputs: 4416
	File system outputs: 32
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
