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

In this chapter, we will be discussing the tools which are used to build Schedulearn. We will also be discussing the issue that Schedulearn is trying to solve and other related works.

Since Horovod has allowed users to their models with one or more graphic cards, the number of waiting time has been reduced by folds. However, it's hard to determine the amount of appropriate resources that a model needs. In most scenarios, users will under or over estimate the amount of resources. This leads to either under or over utilizing the resources.

Schedulearn allows users to predetermine a fixed number graphic cards needed. Moreover, Schedulearn also enables users to set the minimum and the maximum of graphics cards. This allows users to train models with dynamic amount of resources. After the resources required is already determined, the system will algorithmically assign the models to the specified machine in order to maximize its resource utilization. 

## 2.1. Tools

In order to make Schedulearn possible, we have incorporated several most recent open source tools. The tools that we have used are listed below:

### 2.1.1. Containerization 

Containerization is a form of virtualization where applications run in isolated user spaces, called containers, while using the same shared operating system (OS). A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. One of the benefits of containerization is that a container is essentially a fully packaged and portable computing environment. Containers are lightweight and can be easily moved from one machine to another. 

Containers are made from container images. A container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings. These images can be acquired from the docker hub or can be built from scratch per the user's needs.

There are products of containerization, namely Kubernetes, Docker, and Portainer. However, in this thesis, we will be using Docker. Docker is a set of PaaS (Platform as a Service) products that uses OS-level virtualization to deliver software in packages called containers. Docker containers are isolated from one another and bundle their own software, libraries and configuration files; they can communicate with each other through well-defined channels. All containers are run by a single operating-system kernel and are thus more lightweight than virtual machines. Containers are created from images that specify their precise contents. Images are often created by combining and modifying standard images downloaded from public repositories.

### 2.1.2. Distributed Deep Learning

Distributed deep learning is a technique that allows users to speed up training processes using multiple computational graphic cards. There are mainly two types of distributed deep learning techniques: model parallelism and data parallelism. Model parallelism is a technique that allows users to split the model into multiple parts and train them in different GPUs parallelly. Data parallelism is a technique that allows users to split the data into multiple parts and train them in different GPUs parallelly.

In order to make distributed deep learning possible, we have incorporated Horovod. Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make distributed Deep Learning fast and easy to use. Horovod is developed and maintained by Uber AI Labs.

[source](https://towardsdatascience.com/distributed-deep-learning-illustrated-6256e07a0468)

### 2.1.3. RESTful API

RESTful API was first introduced by Roy Fielding in his doctoral dissertation in 2000. REST stands for Representational State Transfer. RESTful API is an architectural style for providing standards between computer systems on the web, making it easier for systems to communicate with each other. RESTful API is a software architectural style that defines a set of constraints to be used for creating web services. Web services that conform to the REST architectural style, called RESTful web services, provide interoperability between computer systems on the internet. RESTful web services allow the requesting systems to access and manipulate textual representations of web resources by using a uniform and predefined set of stateless operations. Other kinds of web services, such as SOAP web services, expose their own arbitrary sets of operations.

FastAPI is a modern, fast (high-performance), web framework for building RESTful API with Python 3.6+ based on standard Python type hints.

The main features are as follows:
* Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
* Fast to code: Increase the speed to develop features by about 200% to 300%.
* Fewer bugs: Reduce about 40% of human (developer) induced errors.
* Intuitive: Great editor support. Completion everywhere. Less time debugging.
* Easy: Designed to be easy to use and learn. Less time reading docs.
* Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
* Robust: Get production-ready code. With automatic interactive documentation.
* Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

## 2.2. Related Work


# 3. Methodology

In this chapter, we will be discussing the issues that some deep learning researchers have been experiencing, especially when it comes to training they humongous models with millions of data. From there, we will discuss on how the proposed system will be able to solve the aforementioned issue.

Previously in Zong-Zuo's work, he has proposed a system, which is called Vodascheduler, that allows users to train their models with one or more graphic cards build on top of Kubeflow. However, there are several issues that users have been experiencing when it comes to training their models with one or more graphic cards.

## 3.1. Motivation

First, Vodaschedulearn is made on top of Kubeflow, thus it uses Kubernetes as the container orchestration tool. Kubernetes is a container orchestration tool that allows users to manage their containers. However, Kubernetes is not the only container orchestration tool that is available. There are other container orchestration tools such as Docker Swarm, Mesos, and Nomad. Most imporantly, Kubernetes requires users to have wide range of knowledge, and thus it is not suitable for developing prototypes such as Schedulearn.

Besides that, it is hard to setup Vodascheduler since Kubernetes needs to be installed in each machine, create a Kubernetes cluster, and install Kubeflow in the cluster. Before they even get to install Kubeflow in each node, they still need to install several third party framework such as Helm. This is not suitable for users who just want to try out the system. In addition to that, it takes at least 3 minutes just for the system to run properly without error. The reason is the system depends heavily on MongoDB which requires some time to start up. Not only MongoDB does take some time to start up, MongoDB docker image is also quite big, and it requires at least 500 MB of storage just to keep MongoDB running with an empty database.

In addition to those issues, users have to go through tedious steps to run their models. Before uploading models to Vodascheduler, users are required to write their own Docker files, and upload their customized Docker files to Docker Hub. Then, users need to write a YAML file to specify the Docker image and how much resources that their models need, as well as the commands that need to be run in the pod. To most users, these steps are tedious and time consuming, and users can not focus on developing their models.

Lastly, it takes a huge amount of time maintaining Vodascheduler. The reason is any changes made, the whole system needs to be restarted for the changes to take effect. Thus, the maintainers of Vodascheduler would have to restart the system several times a day, and it would accumulate to several hours a week.

## 3.2. Proposed System

Schedulearn is created due to the frustrations that we have experienced when we were trying to train our models with Vodascheduler. Schedulearn is a system that allows users to train their models with one or more graphic cards, and each model will be assigned to an avaiable machine algorithmically.

To address the complexities caused by Kubernetes, Schedulearn uses Docker which is lightweight and easy to setup. Since Schedulearn is written in Python, it uses Python's Docker API created by Docker Inc. itself, and it has a well-written documentation as well. In addition to that, docker containers can be created, paused, and deleted programmatically using Python. As a result, less prerequisites knowledge is required to setup Schedulearn.

As for the database, Schedulearn uses SQLite which is a lightweight database framework, and it's already embedded in Python. SQLite does not require any docker container to run since all data is stored in a single file, and the file is 16666 times smaller than the size of a running MongoDB container.

# 4. Analysis


# 5. Conclusion