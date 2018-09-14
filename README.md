Probabilistic Movement Primitive Library
========================================

A Probabilistic Movement Primitive (ProMP) is a probabilistic generative model used to model movement, it is typically
used in the robotics community to learn movements from a human demonstrator (or teacher) and replicate those
movements in a robotic platform.

This repository contains the implementation in Python and C++ of the Probabilistic Movement Primitive framework as
described in [this paper](https://arxiv.org/pdf/1808.10648.pdf). Typically, the operations we want from a ProMP
are:

1) Learning a ProMP from several human demonstrations. Typically, we consider we learn from trajectories in joint 
space.
2) Conditioning in joint space. For example, force the movement to start in the current position of the robot.
3) Conditioning in task space. For example, conditioning a table tennis strike movement to hit the ball in a
certain position in Cartesian coordinates.
4) Controlling a robot to track a ProMP

We provide code for the first three operations. We assume that the learning is done in Python, and only implemented
the adaptation operators in C++ (They are also provided in Python). We also provide code for the following operations:

* Compute the likelihood of a given trajectory for a given ProMP
* Sample trajectories from a ProMP
* Save and load ProMPs

Publications
------------

The implementations provided in this repository are based on the following publications:

1) [Adaptation and Robust Learning of Probabilistic Movement Primitives](https://arxiv.org/pdf/1808.10648.pdf)
2) [Using probabilistic movement primitives for striking movements, IEEE RAS International 
Conference on Humanoid Robots, 2016](https://ieeexplore.ieee.org/abstract/document/7803322/)

Please refer to these papers to understand our implementation, get general information about
probabilistic movement primitives and see the evaluation of the implemented methods in real
robotic platforms.


