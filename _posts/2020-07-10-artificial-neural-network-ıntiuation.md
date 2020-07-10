---
layout: post
author: Batuhan Edgüer
title: Artificial Neural Network Intiuation
date: 2020-07-10T04:50:02.698Z
thumbnail: /assets/img/posts/python.png
category: Python
summary: Theory behind ANN.
---
### Hello everybody. Today I want to talk about theory behind ANN.

I want to divide a few sections this thread;

* **What is a neuron, and how can we implement it?**
* **The Activation Function**
* **Practical Applications**
* **How Neural Network Learn**

I want to start with neurons first, after all, we have the goal of building an artificial neural network.

## **1. The Neuron**

The neuron that forms the base of all neural networks is an approximation of what is seen in the human brain.

![Neuron 1](/assets/img/posts/neuron1.png "Figure 1: Neuron")

This strange rose creature is just one of the thousands who swim within our brains.

The branches around it called dendrites and the tails, which are called axons, are connected to other neurons.

Strangely enough, at the moment a signal is passed between an axon and dendrite, the two don’t actually touch.

Among them, there is a gap. To continue its journey, the signal must act like a stuntman jumping across a deep canyon on a dirtbike. This jump process of the signal passing is called the synapse. For simplicity’s sake, this is the term I will also use when referring to the passing of signals in our Neural Networks.

* **How has the biological neuron been reimagined?**

![Neuron 2](/assets/img/posts/neuron2.png "Figure 2: Neural Network Diagram")

As you can see in the diagram, the signals on the left go to the middle neuron. In humans, these are like touch, smell and sight.

These inputs are **independent variables** in your Neural Network. We are heading through the synapses, passing into the broad green circle, and appearing as **output values** on the other side. For the most part, it is a like-for-like operation. 

The biggest difference between the biological cycle and its artificial equivalent is the level of control, you exert over the input values; on the left hand, the independent variables.

* **You can determine what variables will enter your Neural Network**

It's crucial to remember; either standardizing or normalizing the values of the independent variables. These processes keep your variables within a similar range, making it easier for your Neural Network to process them. This is essential for the operational capability of your Neural Network.

* **Weights**

Each synapse is assigned a weight.

Just as the tautness of the tightrope is integral to the stuntman’s survival, so is the weight assigned to each synapse for the signal that passes along it.

Weights are how Neural Networks learn based on each weight. The weight determines which signals get passed along or not, or to what extent a signal gets passed along.

* **Activation Function**

I want to divide five-part this section.

1. **The Threshold Function**

![TF](/assets/img/posts/tf.png "Figure 3: Threshold Function")

It is a yes or no, black or white, binary function.

2. **Sigmoid Function**

![Sigmoid](/assets/img/posts/sigmoid.png "Figure 4: Sigmoid Function")

Shaplier than its rigid cousin, the Sigmoid’s curvature means it is far better suited to probabilities when applied at the output layer of your Neural Network. If the Threshold tells you the difference between 0 and 1 dollar, the Sigmoid gives you that as well as every cent in between.

3. **Rectifier Function**

![ReLU](/assets/img/posts/relu.png "Figure 5: ReLU")

According to equation 1, the output of ReLu is the maximum value between zero and the input value. An output is equal to zero when the input value is negative and the input value when the input is positive.
For more details, [you can check here.](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

* **How does it work?**

This will be a step-by-step examination of how Neural Networks can be applied by using a real-world example. Property valuations.

In this simple example we have four variables:

1. Area (feet sq),
2. Number of bedrooms,
3. Distance to city (miles),
4. Age of property.

![Example](/assets/img/posts/house_price.png "Figure 6: Real-world example")

Their values go straight to the output layer through the weighted synapses. All four will be analyzed, there will be 
an activation function, and the results will be produced. 

On a basic level, that's comprehensive enough. 
But there's a way to amplify the Neural Network's power and 
increase its accuracy through a very simple addition to the system.

* **Power up**

A hidden layer that sits between the input and output layers can be implemented.

![Power_up](/assets/img/posts/power_up.png "Figure 7: ANN figure")

From this new arrow cobweb, representing the synapses, you begin to understand how those factors cast a wider net of possibilities in different combinations. The result is a much more detailed composite picture of what you seek.

We start with the four variables in the middle of the hidden layer at the left and the top neurons. All four variables are all synapsed to be connected to the neuron. The synapses are not all weighted. They'll either have a value of 0 or a value different than 0. The former indicates importance while the latter means discarding them. In this case, it is common for larger homes to become cheaper on the property market, the further they come from the city.

Distance and Area criteria are met, the neuron applies an activation function and makes its own calculations. The next neuron down may have weighted synapses for Area, Bedroom, and Age. It may have been trained up in a specific area or city where there is a high number of families. New properties are often more expensive than old ones. The way these neurons work and interact means the network itself is extremely flexible, allowing it to look for specific things.

What a journey… If you have any comments or questions about this topic, please comment or contact me. Yours sincerely, Batuhan.

[Source](https://www.udemy.com/course/deeplearning/)