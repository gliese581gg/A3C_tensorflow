#Asynchronous Methods for Deep Reinforcement Learning

(Version 1.0, Last updated :2016.12.12)

###1. Introduction

This is tensorflow implementation of 'Asynchronous Methods for Deep Reinforcement Learning'.(https://arxiv.org/abs/1602.01783)


###2. Usage

    python run.py (args)

    where args :

    -log (log directory name) : Tensorboard event file will be crated in 'logs/(your_input)/' (default : 'A3C')
    -net (A3C or AnDQN) : Network type (A3C or Asynchronous n-step DQN)
    -ckpt (ckpt file path) : checkpoint file (including path)
    -LSTM (True or False) : whether or not use LSTM layer


###3. Requirements:

- Tensorflow
- opencv2
- Arcade Learning Environment ( https://github.com/mgbellemare/Arcade-Learning-Environment )

###4. Test results on 'Pong'
![alt tag](https://github.com/gliese581gg/A3C_tensorflow/blob/master/screenshots/A3CFF.PNG)

Result for Feed-Forward A3C (took about 12 hours, 20 million frames)


![alt tag](https://github.com/gliese581gg/A3C_tensorflow/blob/master/screenshots/A3CLSTM.PNG)

Result for LSTM A3C (took about 20 hours, 28 million frames)

AnDQN is not tested yet!

###5. Changelog

-2016.12.12 : First upload!
