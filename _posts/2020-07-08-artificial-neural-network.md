---
title: Artificial Neural Network
layout: post
author: Batuhan Edgüer
category: Python
date: '2020-07-08 12:19:20 +0000'
summary: Creating Artificial Neural Network with example
thumbnail: "/assets/img/posts/python.png"
---

### Hello everybody. Today I want to talk about ANN model example. 

##### First of all, I will not give you theoretical information about ANN in this post. For this post, you just need to know the algorithm and entry-level artificial intelligence and also you should have to install Anaconda or any python provider like pycharm or etc. [(Source code link.)](https://github.com/BestSithInEU/Artificial_Neural_Network_Example)

There is a data about "Awesome Bank" and firstly I would explain this the issue.
Bank is seen unusual churn rates (Customers leaving at unusually high rates). And they wanted to hire you and understand & solve this problem. When you first look at the data set, you may have difficulty understanding and so I will try to explain by giving examples as much as I can. Awesome Bank decided to create a test group, and there are ten thousand irrelevant people in this test group. But everything is fine, the bank has followed these customers for six months. And as a result, he shared some information of these customers. But what is this information?

1.  CostermId,
2.  Surname,
3.  CreditScore,
4.  Geography,
5.  Gender,
6.  Age,
7.  Tenure (Number of years they've been with the bank.),
8.  Balance,
9.  NumOfProducts (Saving account, credit card, loan, etc.),
10.  HasCrCard (Has credit card)(Boolean),
11.  IsActiveMember (Measure with logged onto their banking, they did a transaction, etc.)(Boolean),
12.  EstimatedSalary,
13.  Exited (After six months who of those customers left) (Boolean).

There is a preview of the data set;

<hr />
<div class="responsive-table">
<table>
      <thead>
        <tr>
          <th scope="col">#</th>
          <th scope="col">RowNumber</th>
          <th scope="col">CustomerId</th>
          <th scope="col">Surname</th>
          <th scope="col">CreditScore</th>
          <th scope="col">Geography</th>
          <th scope="col">Gender</th>
          <th scope="col">Age</th>
          <th scope="col">Tenure</th>
          <th scope="col">Balance</th>
          <th scope="col">NumOfProducts</th>
          <th scope="col">HasCrCard</th>
          <th scope="col">BalIsActiveMemberance</th>
          <th scope="col">EstimatedSalary</th>
          <th scope="col">Exited</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">1</th>
          <td>1</td>
          <td>15634602</td>
          <td>Hargrave</td>
          <td>619</td>
          <td>France</td>
          <td>Female</td>
          <td>42</td>
          <td>2</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>101348</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>

<hr />


###### Those who want to look at the data set in more detail can look through excel or google sheets.

We finally got to the fun side. First of all, we need to change this data set a little. For example, we have no business with the RowNumbers, CustomerId, and Surname. Columns from CreditScore to EstimatedSalary are our features. The Exited part is our dependent variable. But how do we explain them to our computer?

Firstly, lets import our libraries;
{% highlight ruby %}
import numpy as np
import pandas as pd
import tensorflow as tf
{% endhighlight %}
If you don't have these libraries, you can install with these steps;

* Open Anaconda Prompt,
* For numpy;
{% highlight ruby %}
conda install numpy
{% endhighlight %}
*  For pandas;
{% highlight ruby %}
conda install pandas
{% endhighlight %}
* For tensorflow,
{% highlight ruby %}
pip install tensorflow
{% endhighlight %}
* For sklearn
{% highlight ruby %}
conda install scikit-learn
{% endhighlight %}
* For Keras
{% highlight ruby %}
pip install keras
{% endhighlight %}


Or you can install these via Anaconda Navigator, sample video;

<iframe width="560" height="315" src="https://www.youtube.com/embed/8JPvHI8tLPc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


I want to divide the next part into four main headings

## 1. Data Preprocessing.
*  Importing Libraries
{% highlight ruby %}
import numpy as np
import pandas as pd
import tensorflow as tf
{% endhighlight %}

* Importing the dataset
{% highlight ruby %}
dataset = pd.read_csv('data.csv') # Reads the data
X = dataset.iloc[:, 3:-1].values # Taking features from data ( Column 3 to 12 (12 included))
y = dataset.iloc[:, -1].values # Taking dependent variable from data (Last column)
{% endhighlight %}
#####  For those who do not understand ":" and "-1". Firstly python starts counting from zero. ":" means whole row, and "-1" means till last column for our example. (something[row][column] or something[row, column])
Let's have a look our new dataset (X) and dependent variable (y).
{% highlight ruby %}
print(X)
> [619 'France' 'Female' ... 1 1 101348.88]
print(y)
> [1 0 1 ... 1 1 0]
{% endhighlight %}


As I mentioned, the computer does not understand "Geography" and "Gender" sections. So let's turn them into the format they will understand.

*  Label Encoding the "Gender" column
{% highlight ruby %}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
{% endhighlight %}

Let's have a look our dataset again.
{% highlight ruby %}
print(X)
> [619 'France' 0 ... 1 1 101348.88]
{% endhighlight %}

So our function encoded the **female** as "0" randomly.

* One Hot Encoding the "Geography" column
{% highlight ruby %}
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
{% endhighlight %}

Wait a second, why we need to do this one? Answer is very simple actually. If we did label encoding, our data will be like this;

{% highlight ruby %}
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
> [[619 0 0 ... 1 1 101348.88]
   [608 2 0 ... 0 1 112542.58]
	          ...
   [772 1 1 ... 1 0 92888.52]
{% endhighlight %}
The problem here is, since there are different numbers in the same column, the model will misunderstand the data to be in some kind of order, 0 < 1 < 2. But this isn’t the case at all. To overcome this problem, we used One Hot Encoder.
Again check the data set with one hot encoding. And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

{% highlight ruby %}
print(X)
> [[1.0 0.0 0.0 ... 1 1 101348.88]
   [0.0 0.0 1.0 ... 0 1 112542.58]
   [1.0 0.0 0.0 ... 1 0 113931.57]
{% endhighlight %}

So... France encoded with [1, 0, 0], Spain encoded with [0, 0, 1], and finally Germany [0, 1, 0].
* Splitting the dataset into the Training set and Test set (Because we don't have any Test set...)

{% highlight ruby %}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
{% endhighlight %}

This code create X_train, y_train and X_test, y_test. We had to do this. Because we had no test set, we would not know our training result and the ratio of 0.2 can be changed but 0.2 is generally used.

*  Feature Scaling
{% highlight ruby %}
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
{% endhighlight %}

This feature scaling is so fundamental and we should apply to all our features.

## 2.  Building the ANN

* Initializing the ANN
{% highlight ruby %}
ann = tf.keras.models.Sequential()
{% endhighlight %}

We create  this variable for creating fully connected network.

* Adding the input layer and the first hidden 
{% highlight ruby %}
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
{% endhighlight %}

We create six neurons. But this six comes from actually nowhere, just experimentation. "relu" means activation function is rectifier, and activation function in the hidden layer of fully connected neural network must be rectifier activation function. And "dense" class for fully connection between neurons.

* Adding the second hidden layer
{% highlight ruby %}
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
{% endhighlight %}
* Adding the output layer
{% highlight ruby %}
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
{% endhighlight %}
If we look dependent variable again which is "Exited, since we want to predict binary variable (zero or one) we only need one neuron.  And activation function of the output layer should be sigmoid activation function.
![Sigmoid vs ReuLU](https://miro.medium.com/max/1452/1*XxxiA0jJvPrHEJHD4z893g.png)
##### Figure 1: Sigmoid vs ReuLU

## 3.  Training the ANN

* Compiling the ANN
{% highlight ruby %}
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
{% endhighlight %}
Parameters: 
	1. Optimizer: Choosing optimizer,
	2. Loss: Choosing loss function (For binary outcome, loss function should be binary crossentropy,	
	3. Metrics.
	
* Training the ANN on the Training set
{% highlight ruby %}
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
{% endhighlight %}
Batch learning more efficient and more performant when training ANN. 
After executing the code, we should see this;
{% highlight ruby %}
...
Epoch 99/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3352 - accuracy: 0.8648
Epoch 100/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3355 - accuracy: 0.8646
{% endhighlight %}
So our general accuracy is 86.46%. It's really good actually.

## 4. ## 4. Data Preprocessing.

* Predicting the Test set results
{% highlight ruby %}
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) # We add this line for rounding.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
> [[0 0]
   [0 1]
   [0 0]
    ...
   [0 0]
   [0 0]
   [0 0]]
{% endhighlight %}
Left of the vector shows prediction, and right of the vector shows real results

*  Making the Confusion Matrix
{% highlight ruby %}
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
> [[1514   81]
   [ 198  207]]
> 0.8605
{% endhighlight %}
So 1514 correct predictions that the customer stay in the bank and 207 correct predictions that the customer leaves the bank. So our general accuracy is 86.05%

What a journey... If you have any comments or questions about this topic, please comment or contact me.
Yours sincerely, Batuhan.

[Source](https://www.udemy.com/course/deeplearning/)