# Shakespearean Text Generation

 
#### Sources & Articles
#### https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_5.pdf
#### https://juejin.cn/post/6995777526308012069
#### https://clownote.github.io/2020/08/20/DeepLearningWithPython/Deep-Learning%20with-Python-ch8_1/
#### https://blog.csdn.net/weixin_46489969/article/details/125525879
 
 
#### Build 371778 samples by using the 1115394 chars in the text
####                                          feature       label
#### 1 Sample:  60 characters -> 1 character
 
#### Input: one-hot encoding (no embedding layer), 39 values [ 0 0 0 0 … 1 .. 0 0 0 ]
 
#### input_shape=(60, 39) # 39 input values for each call and call 60 times - 1 sample
 
#### LSTM Layer: 128 status values (128 neurons)
#### Parameters:  (39 W + 128 W + 1b) x 128 x 4 = 86016  # 4 times of the RNN's parameters
 
#### SoftMax Layer: 39 probability values
#### Parameters: (128 W + 1 b) x 39 = 5031
#### Prediction: probability -> next char, one-hot by using  Multinomial Distribution.

#### 1 Epoch without GPU:
![image](https://github.com/yinanericxue/Shakespearean-Text-Generation/assets/102645083/369878b5-84ef-46d9-84fa-c119ce274e44)

#### 1 Epoch with GPU:
![image](https://github.com/yinanericxue/Shakespearean-Text-Generation/assets/102645083/863f0cb4-34e1-4b21-8aaa-0ad91095c8cf)

#### Binomial Distribution - flip a coin
#### https://www.investopedia.com/terms/b/binomialdistribution.asp
 
#### Multinomial Distribution - roll a dice
#### https://www.statisticshowto.com/multinomial-distribution/
#### https://en.wikipedia.org/wiki/Multinomial_distribution

#### numpy.random.multinomial
#### https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html

#### numpy.argmax
#### https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
 
#### Old
 
#### 5000 unique words；
#### 1 sentence： 500 words，32 values per word
 
#### LR ( 一次输入500个word)
#### Embedding layer: 5000 x 32 = 160000 parameters
#### LR Layer: 500x32 + 1 = 16,001 parameters
 
#### RNN ( 一次输入1个word，需要500次 )
#### Embedding layer: 5000 x 32 = 160000 parameters
#### RNN Layer: 32 x ( 32 + 32 + 1 ) = 2,080 parameters,
 
#### LSTM
#### Embedding layer: 5000 x 32 = 160000 parameters
##### LR Layer: 4 x 32 x ( 32 + 32 + 1 ) = 8,320 parameters 
