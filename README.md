# Shakespearean-Text-Generation

 
# Char-Level, TF Dev Cert/NLP/generate_text_char_level.py
 
# Word-Level, TF Dev Cert/NLP/generate_text_word_level.py
 
 
#################################################### Example: Char-Level

RNN模型与NLP应用(6/9)：Text Generation (自动文本生成)
https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_5.pdf
 
https://juejin.cn/post/6995777526308012069
https://clownote.github.io/2020/08/20/DeepLearningWithPython/Deep-Learning%20with-Python-ch8_1/
https://blog.csdn.net/weixin_46489969/article/details/125525879
 
 
##### run "TF Dev Cert/NLP/generate_text_char_level.py"
 
Build 371778 samples by using the 1115394 chars in the text
                          feature               label
1 Sample:  60 characters -> 1 character
 
Input: one-hot encoding (no embedding layer), 39 values [ 0 0 0 0 … 1 .. 0 0 0 ]
 
input_shape=(60, 39) # 39 input values for each call and call 60 times - 1 sample
 
LSTM Layer: 128 status values ( 128 neurons )
Parameters:  (39 W + 128 W + 1b) x 128 x 4 = 86016  # 4 times of the RNN's parameters
 
SoftMax Layer: 39 probability values
Parameters: (128 W + 1 b) x 39 = 5031
Prediction: probability -> next char,  one-shot by using  Multinomial Distribution.
1 Epoch without GPU:
1 Epoch with GPU:

 
 
 
 

 

 
 
 
 

 
 
 
# Bernouli Distribution



 
 
 
伯努利分布：
对于随机变量x，P(x = 1) = p，实验成功；P(x = 0) = 1-p，实验失败。
 
 
 
# Binomial Distribution - flip a coin
https://www.investopedia.com/terms/b/binomialdistribution.asp
 
 

 
 
二项分布，就是进行n次伯努利实验（单次成功的概率为p），k次成功的概率：

 

 

 
 
 
# Multinomial Distribution - roll a dice
https://www.statisticshowto.com/multinomial-distribution/
https://en.wikipedia.org/wiki/Multinomial_distribution
 
 

 
多项分布：
单次实验，结果有多个（扔骰子，就是6个）；
单次实验的概率为： p1, p2, p3, p4, p5, p6；总和为1。
现在进行n次实验，对应的出现次数为：x1, x2, x3, x4, x5, x6；总合为n；概率为：

 
 
# numpy.random.multinomial
https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
 
 

 
 
# numpy.argmax
https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
 

 
 
 
 
 
#### Old
 
5000 unique words；
1 sentence： 500 words，32 values per word
 
# LR ( 一次输入500个word)
Embedding layer: 5000 x 32 = 160000 parameters
LR Layer: 500x32 + 1 = 16,001 parameters
 
# RNN ( 一次输入1个word，需要500次 )
Embedding layer: 5000 x 32 = 160000 parameters
RNN Layer: 32 x ( 32 + 32 + 1 ) = 2,080 parameters,
 
# LSTM (  一次输入1个word，需要500次 )
Embedding layer: 5000 x 32 = 160000 parameters
LR Layer: 4 x 32 x ( 32 + 32 + 1 ) = 8,320 parameters
 
 
 
# CHN
https://zhuanlan.zhihu.com/p/31656392
https://blog.csdn.net/newlw/article/details/122546868
 
 
 
 
 
 
 
 
#################################################### Example: Word-Level
#################################################### Example: Word-Level
https://www.kaggle.com/datasets/aashita/nyt-comments
https://www.kaggle.com/datasets/aashita/nyt-comments?resource=download
https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms
 
 
利用LSTM，学习NYT的新闻标题，然后生成新闻标题。
先利用Teras Tokenizer，向量化文库，采用数字序列替代文本。
利用NGRAM，左填充，生成样本，18个词（Index序列） -> 1个词（One-Hot表示）。
建立LSTM模式时，要增加Embedding Layer。
 
 
 
 
 
##### run "TF Dev Cert/NLP/generate_text_word_level.py"
 
 

 
 
831 headlines with 2421 unique words ( ID: 1 ~ 2421)；
ID: 0 is reserved, so totally 2422 words;
 
Generate 4806 samples:
Input: Sequence of ID, 18 IDs ( including ID:0  )
Output: One-hot, 2422 classes )
 
 

 
 
 tokenizer.fit_on_texts(corpus)
一共2421个字，ID为1~2421；ID 0，被预留。
所以把长度定义为2421，ID 从 0 到2421。
 
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

 
 
 
 
 
 
 
 
 
# keras.preprocessing.text.Tokenizer
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
 

 
 
向量化一个文库：
建立一个字典，为每个词提供一个ID（Index），0被预留。用一组ID，表达文本。
去掉所有的标点符号。
 
 
 
 
# fit_on_texts
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#fit_on_texts
 

 
 
 
 
 
# ngram
https://www.mathworks.com/discovery/ngram.html
 

 
 
# pad_sequences
https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
 

 
 
 
# to_categorical
https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
 

 
 
 
# Embedding Layer
https://www.tensorflow.org/text/guide/word_embeddings#:~:text=An%20embedding%20is%20a%20dense,weights%20for%20a%20dense%20layer
https://keras.io/api/layers/core_layers/embedding/
 
https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce
 

 

 
 
 
#################################################### Summary
#################################################### Summary
 
 

 
 
 
 
 
 
 
 
 
 
#################################################### NLTK - Natural Language Toolkit
#################################################### NLTK - Natural Language Toolkit
https://www.nltk.org/
 

 
 
#################################################### Example: Word-Level
#################################################### Example: Word-Level
https://towardsdatascience.com/word-and-character-based-lstms-12eb65f779c2
https://github.com/ruthussanketh/natural-language-processing/blob/main/word-and-character-LSTM/corpus.txt
 
https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb
 
 
https://towardsdatascience.com/text-generation-gpt-2-lstm-markov-chain-9ea371820e1e
 
 
 
https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms
 
https://www.mathworks.com/help/deeplearning/ug/word-by-word-text-generation-using-deep-learning.html
 
 
 
 
 
 
#################################################### Example: Char-Level (Tensorflow官方)
#################################################### Example: Char-Level (Tensorflow官方)
https://www.tensorflow.org/text/tutorials/text_generation
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 
https://towardsdatascience.com/text-generation-using-rnns-fdb03a010b9f
https://www.analyticsvidhya.com/blog/2022/02/explaining-text-generation-with-lstm/
 
 
 

 
 
 
 
# ln
https://www.medcalc.org/manual/ln-function.php

 

 
 
# Logit in Math
 
https://en.wikipedia.org/wiki/Logit
https://deepai.org/machine-learning-glossary-and-terms/logit
https://lucasdavid.github.io/blog/machine-learning/crossentropy-and-logits/
 
 

 

 

 

 
Sigmod:
 

 

 
 
# Logit in Machine Learning
https://developers.google.com/machine-learning/glossary/#logits
https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
 
https://www.cnblogs.com/SupremeBoy/p/12266155.html
 

 
 
数学中的Logit 与 机器学习中的Logit 不一样。
机器学习中的Logit，就是WX + b，没有激活函数。这样就可以将Softmax与cross-entropy一起实现。

 

 
 
 
 
# from_logits
https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
 

 
 
#tf.function
 
 
 
 

 
 

 

 
 
# RNN many-to-many loss
https://www.cnblogs.com/tangweijqxx/p/10637396.html
https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/12/09/01-RNN-Many-to-many.html

 

 
 
# Ragged Tensor
https://www.tensorflow.org/guide/ragged_tensor
 

 
 
# tf.strings.unicode_split
https://www.tensorflow.org/api_docs/python/tf/strings/unicode_split
 
 
 
 
# tf.keras.layers.StringLookup 
https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup?version=nightly
 
 /TF Advanced/ProprocessingWithStatus.py
 
 
# tf.data.Dataset.from_tensor_slices
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#from_tensor_slices
 
 
# dataset map
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#map

 
 
 
 
 
b'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '
 
b'are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you k'
 
b"now Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\nFirst Citizen:\nLet us ki"
 
b"ll him, and we'll have corn at our own price.\nIs't a verdict?\n\nAll:\nNo more talking on't; let it be d"
 
b'one: away, away!\n\nSecond Citizen:\nOne word, good citizens.\n\nFirst Citizen:\nWe are accounted poor citi'
 
 
 
每个样本：
0~99，1 ~100
 
 
 
1个Segment：
 
输入 60 x 57， 输出 57
 
1个样本： 输入 100 x 66，输出 100 x 66
1个Batch：64个样本；
 
 
 
256
