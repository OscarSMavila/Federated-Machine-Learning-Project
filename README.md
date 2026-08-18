Enhancing Model Aggregation in Federated Machine Learning

A federated learning project exploring whether trust-based model aggregation and model pruning can improve training speed, accuracy, and communication efficiency.

Overview

Federated Learning (FL) allows multiple clients to train a shared machine learning model without directly sharing their training data.

This project investigates Behavioral Trust-Aware Federated Averaging (BTA-FedAvg), a modified version of FedAvg that gives more influence to clients that consistently perform well.

The project also explores model pruning to reduce redundant model parameters before models are sent to the server.

Approach

The project uses the Flower federated learning framework to simulate up to 50 clients.

The main components are:

FedAvg — Standard federated averaging baseline
AdaptiveFedAvg — Adaptive client weighting
FLEvaluate — Additional aggregation baseline
BTA-FedAvg — Proposed trust-aware aggregation strategy
Model pruning — Tested in combination with federated learning
BTA-FedAvg

Each client receives a trust score based on its historical accuracy. The score is updated using an exponential moving average.

Clients with consistently better performance receive a greater weight during aggregation, while lower-performing clients have less influence.

A 15-round warmup period is used before trust scores begin affecting aggregation.

Models and Datasets
Model
EfficientNet-B0
Pretrained ImageNet weights
Images resized to 224×224
Datasets
CIFAR-100
SVHN

The datasets were partitioned among simulated clients using stratified splits.

Training

The main experiments used:

50 simulated clients
100 federated rounds
Adam optimizer
Learning rate: 0.001
Batch size: 128
Client-side pruning: 20%
Flower federated learning simulation
Results
CIFAR-100
Metric	FedAvg	AdaptiveFedAvg	BTA-FedAvg	FLEvaluate
Accuracy	85.70%	6.12%	85.41%	85.67%
F1 Score	85.66%	4.83%	85.36%	85.61%
Convergence Round	82	101	57	74
Processing Time	18,013s	11,059s	17,883s	25,144s

BTA-FedAvg achieved the fastest convergence, reaching its target accuracy in 57 rounds compared with 82 rounds for standard FedAvg.

AdaptiveFedAvg performed significantly worse in this experiment.

SVHN
Metric	BTA-FedAvg	FedAvg	BTA-FedAvg + Pruning
Accuracy	97.39%	97.40%	97.47%
F1 Score	97.39%	97.40%	97.47%
Convergence Round	79	89	93
Processing Time	19,620s	19,787s	20,405s

Adding pruning resulted in a small improvement in accuracy, but increased the number of rounds required to converge and increased processing time.

Key Findings
BTA-FedAvg converged faster than standard FedAvg in the tested experiments.
Trust-based weighting can give consistently performing clients more influence during aggregation.
20% pruning produced a small accuracy improvement on SVHN.
Pruning introduced additional training time and slower convergence.
AdaptiveFedAvg performed poorly on the CIFAR-100 experiment.
BTA-FedAvg and FedAvg achieved similar final accuracy, while BTA-FedAvg generally converged faster.
Limitations

The current implementation has several limitations:

Clients must use the same model architecture for direct weight aggregation.
Trust is currently based primarily on historical client accuracy.
Pruning requires careful tuning to avoid negatively affecting training.
The experiments use simulated clients rather than physical edge devices.
More testing is needed with unreliable or malicious clients.
Future Work

Potential improvements include:

Adding additional trust metrics such as participation consistency and gradient alignment.
Testing the system with malicious or unreliable clients.
Further tuning pruning thresholds.
Supporting heterogeneous model architectures.
Exploring knowledge distillation for communication between different model architectures.
Comparing against additional robust federated learning strategies.
Technologies
Python
PyTorch
Flower
NumPy
TensorFlow/Keras
EfficientNet-B0
CIFAR-100
SVHN
Authors

Oscar Mavila
Joseph Woods

University of Colorado Colorado Springs
