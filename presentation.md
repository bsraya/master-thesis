---
marp: true
theme: uncover
math: mathjax
header: 'Schedulearn: An Elastic Learning Platform Using Microservices'
paginate: true
---

# Schedulearn

***An Elastic Learning Platform Using Microservices***

Author: Bijon Setyawan Raya
Advisor: Che-Rung Lee

---

# Outline

* Introduction
* Motivation
* Background
* Design & Implementation
* Experimentation
* Conclusion
* Future Work

---

# Introduction

---

### Introduction

#### _Objectives_

* To implement a lightweight deep learning distributed scheduling system

* To implement different scheduling algorithms

* To implement job migrations

* To compare the performance of scheduling algorithms

---

### Introduction

#### _Contributions_

1. A lightweight deep learning scheduling system

2. A customizeable scheduling system

3. Implemented job migrations

---

# Motivation

---

# _Why Schedulearn?_

--- 

_Why Schedulearn?_

### 1. Monolithic systems are hard to customize and to scale.

---

_Why Schedulearn?_

### 2. Why use Kubernetes?

---

_Why Schedulearn?_

### 3. Why use Golang?

---

# Background

---

### Containerization

![h:500](https://www.docker.com/wp-content/uploads/2021/11/container-what-is-container.png)

---

### Distributed Deep Learning

![h:500 w:1000](https://xiandong79.github.io/downloads/ddl1.png)

---

### Web Services

![w:1000](https://miro.medium.com/max/1189/0*ZL3Xc178YqldwFcb.png)

---

### Persistent Storage

![h:500](https://www.holistics.io/blog/content/images/2018/08/dbdiagram.io---diagram-only.png)

---

# Design & Implementation

---

### System Overview

![height:450px](./figures/schedulearn-architecture.png)

---

![bg right height:350px](./figures/procedure.png)

#### How Does It Work?
1. Send
2. Schedule
3. Store
4. Retrieve
5. Train
6. Return result
7. Send back

---

### FIFO

![h:500](./figures/FIFO.png)

---

### Round Robin

![h:500](./figures/round-robin.png)

---

### Job Migrations

Kill + respawn

![h:400](./migration.png)

---

# Experimentation

---

### Testbed Specification

1. 4 x Nvidia 1080 Ti graphics cards
2. Intel Xeon E5-2678 v3, 48 cores
3. 128 GB RAM
4. 10G PCIe network

---

### Speedups

![bg right:50% height:800](./figures/mnist.png)

1. TensorFlow Keras MNIST
2. TensorFlow MNIST
3. PyTorch MNIST
4. Apache MXNet MNIST

---

### Scalability

![height:500](./figures/scalability.png)

---

### Makespan & Turnaround Time

![height:500](./figures/tensorflow_comparison.png)

---

### Makespan & Turnaround Time

![height:500](./figures/pytorch_comparison.png)

---

### Makespan & Turnaround Time

![height:500](./figures/mxnet_comparison.png)

---

### Job Migrations

![height:500](./makespan.png)

---

### Job Migrations

![height:500](./turnaround.png)

---

### Job Migrations

![height:500](./no-of-migration.png)

---

### FIFO + PyTorch

![height:500](./pytorch/fifo/1.png)

---

### FIFO + TensorFlow

![height:500](./tensorflow/fifo/1.png)

---

### FIFO + Apache MXNet

![height:500](./mxnet/fifo/1.png)

---

### RR + PyTorch

![height:500](./pytorch/rr/1.png)

---

### RR + TensorFlow

![height:500](./tensorflow/rr/1.png)

---

### RR + Apache MXNet

![height:500](./mxnet/rr/1.png)

---

### Speedup of Pre-Trained Models

![height:400](./figures/resnet_duration_speedup.png)

---

### Scalability of Pre-Trained Models

![height:400](./figures/resnet.png)

---

# Conclusion

1. FIFO maximizes GPU utilization for each server.
2. Round Robin minimizes GPU utilization for each server.
3. Smaller jobs gain more speedup, while not larger jobs.
4. Job migrations help ensure job completion and stabilize the average turnaround time.

---

# Future Work

---

#### Future Work 

More Scheduling Algorithms

* Shortest-Remaining-Job-First (SRJF)
* Shortest-Remaining-Time-First (SRTF)
* Earliest-Deadline-First (EDF)
* Tiresias
* FfDL
* AFS-L

---

#### Future Work 

A better queuing system

---

#### Future Work

A better training time estimator

* Available resources
* No. of jobs
* Jobs' hyperparameters
* Resource requirements