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

In this chapter, we will be discussing the issue that Schedulearn is trying to solve, as well as the tools that are used to solve the issue and the related works.

## 2.1. Motivations

First, Vodascheduler is made on top of Kubeflow, thus it uses Kubernetes as the container orchestration tool. However, Kubernetes requires users to have wide range of knowledge, and thus it is not suitable for developing prototypes. Moreover, developers with litle or no knowledge of Kubernetes will have a hard time to comprehend the big picture of the system.

Besides that, it is hard to setup Vodascheduler since Kubernetes needs to be installed in each machine. After Kubernetes is installed, users need to create a Kubernetes cluster and install Kubeflow in the cluster. Before they even get to install Kubeflow in each node, they still need to install several third party framework such as Helm and NVIDIA GPU Operator. This is not suitable for users who just want to try out the system. In addition to that, it takes at least 3 minutes just for the system to run properly without throwing any errors. The reason is the system depends heavily on MongoDB which requires some time to start up. Not only MongoDB does take some time to start up, MongoDB docker image is also quite big, and it requires at least 500 MB of storage just to keep MongoDB running with an empty database.

In addition to those issues, users have to go through tedious steps to run their models. Before uploading models to Vodascheduler, users are required to write their own Docker files, and upload their customized Docker files to Docker Hub. Then, users need to write a YAML file to specify the Docker image and how much resources that their models need, as well as the commands that need to be run in the pod. To most users, these steps are tedious and time consuming, and users can not focus on developing their models.

Lastly, it takes a huge amount of time maintaining Vodascheduler. The reason is any changes made, the whole system needs to be restarted for the changes to take effect. Thus, the maintainers of Vodascheduler would have to restart the system several times a day, and it would accumulate to several hours a week.

## 2.2. Tools

Schedulearn is made with a microservice architecture in place. A microservice architecture is an architectural style that structures an application as a collection of loosely coupled services. Each service is a small, independent process that communicates with other processes to achieve a business goal. The microservice architecture is a style of software design that emphasizes building single-purpose services that are independently deployable, scalable, and replaceable. By decomposing Schedulearn into small, independent services, we can build and deploy each service independently. This allows us to scale each service independently. Moreover, it also allows us to replace a service with a different implementation without affecting other services. In order to make Schedulearn possible, we have incorporated several most recent open source tools, namely:

### 2.2.1. Docker 

Docker is a set of PaaS (Platform as a Service) products that uses OS-level virtualization to deliver software in packages called containers. Docker containers are isolated from one another and bundle their own software, libraries and configuration files; they can communicate with each other through well-defined channels. All containers are run by a single operating-system kernel and are thus more lightweight than virtual machines. Containers are created from images that specify their precise contents. Images are often created by combining and modifying standard images downloaded from public repositories. There are products of containerization, namely Kubernetes, Docker, and Portainer. The reason why we choose Docker is because it has already matured, and it is widely used in the industry.

Containerization is a form of virtualization where applications run in isolated user spaces, called containers, while using the same shared operating system (OS). A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. One of the benefits of containerization is that a container is essentially a fully packaged and portable computing environment. Containers are lightweight and can be easily moved from one machine to another. 

Containers are made from container images. A container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings. These images can be acquired from the docker hub or can be built from scratch per the user's needs.

### 2.2.2. Horovod

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make distributed Deep Learning fast and easy to use. With the help of the ring-allreduce approach, Horovod is able to achieve high bandwidth between nodes. Horovod is also able to scale up to thousands of GPUs and to efficiently run a variety of workloads, including image classification, natural language processing, and reinforcement learning. Horovod is developed and maintained by Uber AI Labs.

The ring-allreduce algorithm is each of $N$ nodes communicates with two of its peers $2 ∗ (N − 1)$ times. During this communication, a node sends and receives chunks of the data buffer. In the first $N − 1$ iterations, received values are added to the values in the node's buffer. In the second $N − 1$ iterations, received values replace the values held in the node's buffer.

Distributed deep learning is a technique that allows users to speed up training processes using multiple computational graphic cards. There are mainly two types of distributed deep learning techniques: model parallelism and data parallelism. Model parallelism is a technique that allows users to split the model into multiple parts and train them in different GPUs parallelly. Data parallelism is a technique that allows users to split the data into multiple parts and train them in different GPUs parallelly.

[source](https://towardsdatascience.com/distributed-deep-learning-illustrated-6256e07a0468)

### 2.2.3. FastAPI

FastAPI is a modern, fast (high-performance), web framework for building RESTful API with Python 3.6+ based on standard Python type hints.

RESTful API was first introduced by Roy Fielding in his doctoral dissertation in 2000. REST stands for Representational State Transfer. RESTful API is an architectural style for providing standards between computer systems on the web, making it easier for systems to communicate with each other. RESTful API is a software architectural style that defines a set of constraints to be used for creating web services. Web services that conform to the REST architectural style, called RESTful web services, provide interoperability between computer systems on the internet. RESTful web services allow the requesting systems to access and manipulate textual representations of web resources by using a uniform and predefined set of stateless operations. Other kinds of web services, such as SOAP web services, expose their own arbitrary sets of operations.

The main reason that we incorporate an API into the system is to make it easier for users to interact with the system. Users can interact with the system by sending HTTP requests to the API. The API will then process the request and return the response to the user. Moreover, the API can be easily configured to receive and block requests depending on the state of the system. As a result, any changes in the API will not affect the entire system.

These following reasons are the other reasons why we choose FastAPI as our RESTful API framework:
* Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
* Fast to code: Increase the speed to develop features by about 200% to 300%.
* Fewer bugs: Reduce about 40% of human (developer) induced errors.
* Intuitive: Great editor support. Completion everywhere. Less time debugging.
* Easy: Designed to be easy to use and learn. Less time reading docs.
* Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
* Robust: Get production-ready code. With automatic interactive documentation.
* Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

### 2.2.4. SQLite

SQLite is a relational database management system contained in a C library. In contrast to many other database management systems, SQLite is not a client-server database engine. Rather, it is embedded into the end program. SQLite is ACID-compliant and implements most of the SQL standard, using a dynamically typed, table-based, hierarchical data structure, with optional typing. The source code for SQLite is in the public domain.

## 2.3. Related Work

### 2.3.1. Distributed Deep Learning

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. It utilizes a ring-allreduce method to distribute the training process. The ring-allreduce algorithm is each of $N$ nodes communicates with two of its peers $2 ∗ (N − 1)$ times. During this communication, a node sends and receives chunks of the data buffer. In the first $N − 1$ iterations, received values are added to the values in the node's buffer. In the second $N − 1$ iterations, received values replace the values held in the node's buffer.

### 2.3.2. Elastic Deep Learning

Tsung-Tsuo Hsieh proposed a framework that allows users to train deep learning models with one or more graphic cards. He proposed a method that allows models to be trained parallely which the number of graphic cards can elastically scale up and down depending on the availability of resources in the system. His framework is called Vodascheduler, and it was made on top of Kubeflow and uses Kubernetes as the underlying container orchestration system.

# 3. Design and Implementation

In this chapter, we will be discussing the design and implementation of Schedulearn. We will be discussing the design of the system, the implementation of the system, and the testing of the system.

## 3.1. System Overview

Schedulearn is created due to the frustrations that we have experienced when we were trying to train our models with Vodascheduler.

To address the complexities caused by Kubernetes, Schedulearn uses Docker which is lightweight and easy to setup. Since Schedulearn is written in Python, it uses Python's Docker API created by Docker Inc. itself, and it has a well-written documentation as well. In addition to that, docker containers can be created, paused, and deleted programmatically using Python. As a result, less prerequisites knowledge is required to setup Schedulearn.

As for the database, Schedulearn uses SQLite which is a lightweight database framework, and it's already embedded in Python. SQLite does not require any docker container to run since all data is stored in a single file, and the file is 16,666 times smaller than the size of an empty and running MongoDB container.

# 4. Analysis


# 5. Conclusion