# Abstract

# 1. Introduction

Deep learning has been one of the sought-after fields in the research community. Due to the deep learning algorithms efficiency in inferring information, a great number of researchers from different fields has been using Deep Learning to help them automating a certain task, such as inference or classification.

Deep Learning is one of the machine learning methods based on neural networks with representational learning. Deep learning can be categorized into five problems: learning problems, hybrid learning problems, statistical, inference, and learning techniques. However, two types of problems are going to be the main highlight, namely learning and hybrid leaning problems. Learning problems include supervised, unsupervised, and reinforcement learning. Hybrid learning problems include semi-supervised, self-supervised, and multi-instance learning.

Even though many deep learning algorithms have been more efficient given the number of research have been done to improve them, their efficiencies are still no match to the size of data that is generated everyday. Thus, a new approach of training models was needed to solve this issue. 

In 2018, two engineers from Uber Technologies made a framework that capable of solving the aforementioned issue. They proposed a method that allows models to be trained parallely with one or more graphic cards.

Even though a model can be trained with multiple graphic cards, users still have to manually consider the amount of resources each model requires. Most of the time, users will over or under estimate the amount of resources needed to train a model. 

In this thesis, I will be proposing a system that schedules deep learning models with fixed or dynamic amount resources to any available machines. Most importantly, it enables users to train deep learning models with one or more graphic cards. Since this framework is made to schedule deep learning models, thus it is called Schedulearn.

In chapter 2, we will be discussing the issue that Schedulearn is trying to solve and other related works. In chapter 3, we will be discussing the implementation and the architecture of Schedulearn. In chapter 4, we will be discussing the evaluation from analyzing Schedulearn performance, as well as ideas that can draw from the experiement. In chapter 6, we will summerize the thesis and discuss the future work.

# 2. Background

In this chapter, we will be discussing the frustrations that some deep learning researchers have been experiencing, especially when it comes to training they humongous models with millions of data. From there, we will discuss on how the proposed system will be able to solve the aforementioned issue.

Since Horovod has allowed users to their models with one or more graphic cards, the number of waiting time has been reduced by folds. However, it's hard to predetermine the amount of appropriate resources. In most scenarios, users will under or over estimate the amount of resources they need. This leads to either under or over utilizing the resources.

With Schedulearn, users do not have to worry about assigning the amount of resources needed to train a model. Schedulearn will help users placing their models in a machine where the amount of resources needed is available. Most imporantly, they can focus more on developing models.

Schedulearn allows users to predetermine or algorithmically system assesses the amount of graphic cards required. Moreover, after the resources required is already determined, the system will algorithmically assign the models to the specified machine in order to maximize its resource utilization. 

## 2.1. Tools

Schedulearn is build on top of 

### 2.1.1. Docker 

Docker is a set of PaaS (Platform as a Service) products that uses OS-level virtualization to deliver software in packages called containers. Docker containers are isolated from one another and bundle their own software, libraries and configuration files; they can communicate with each other through well-defined channels. All containers are run by a single operating-system kernel and are thus more lightweight than virtual machines. Containers are created from images that specify their precise contents. Images are often created by combining and modifying standard images downloaded from public repositories.

### 2.1.2. Horovod

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make distributed Deep Learning fast and easy to use. Horovod is developed and maintained by Uber AI Labs.

### 2.1.3. FastAPI

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

The main features are as follows:
* Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
* Fast to code: Increase the speed to develop features by about 200% to 300%. *
* Fewer bugs: Reduce about 40% of human (developer) induced errors. *
* Intuitive: Great editor support. Completion everywhere. Less time debugging.
* Easy: Designed to be easy to use and learn. Less time reading docs.
* Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
* Robust: Get production-ready code. With automatic interactive documentation.
* Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

## 2.2. Related Work



# 3. Methodology and Implementation



# 4. Analysis


# 5. Conclusion