#Codeauschnitte aus Vorlesung von Herrn Prof. Dr. Gepperth HS Fulda
mnistPath = "/Users/mh/PycharmProjects/Forschungsprojekt/venv/include/mnist.pkl.gz"

import matplotlib as mp ;
mp.use("Qt4Agg") ;
import gzip, pickle,numpy as np, matplotlib.pyplot as plt ;
import numpy.random as npr, tensorflow as tf, sys  ;
from matplotlib.widgets import Button ;
import math ;

# expects 1D array
def sm(arr):
  num = np.exp(arr) ;
  den = num.sum() ;
  return num/den ;

def test_cb(self):
    global testit ;
    ax1.cla();
    ax2.cla();
    ax3.cla();
    ax1.imshow(testd[testit].reshape(28,28)) ;
    confs =sm(testout[testit]) ;
    ax2.bar(range(0,10),confs);
    ax2.set_ylim(0,1.)
    ce = -(confs*np.log(confs+0.00000001)).sum() ;
    ax3.text(0.5,0.5,str(ce),fontsize=20)
    testit = testit + 1;
    f.canvas.draw();
    print ("--------------------") ;
    print("logits", testout[testit], "probabilities", sm(testout[testit]), "decision", testout[testit].argmax(), "label", testl[testit].argmax()) ;



sess = tf.Session();

with gzip.open(mnistPath, 'rb') as f:
    ((traind,trainl),(vald,vall),(testd,testl))=pickle.load(f, encoding='bytes')

data_placeholder = tf.placeholder(tf.float32,[None,784]) ;
label_placeholder = tf.placeholder(tf.float32,[None,10]) ;

fd = {data_placeholder: traind, label_placeholder : trainl } ;
#fd = {}

Wh1 = tf.Variable(npr.uniform(-0.01,0.01, [784,200]),dtype=tf.float32, name ="Wh1") ;
bh1 = tf.Variable(npr.uniform(-0.01,0.01, [1,200]),dtype=tf.float32, name ="bh1") ;

Wh2 = tf.Variable(npr.uniform(-0.1,0.1, [200,200]),dtype=tf.float32, name ="Wh2") ;
bh2 = tf.Variable(npr.uniform(-0.01,0.01, [1,200]),dtype=tf.float32, name ="bh2") ;

Wh3 = tf.Variable(npr.uniform(-0.1,0.1, [200,200]),dtype=tf.float32, name ="Wh3") ;
bh3 = tf.Variable(npr.uniform(-0.01,0.01, [1,200]),dtype=tf.float32, name ="bh3") ;

W = tf.Variable(npr.uniform(-0.01,0.01, [200,10]),dtype=tf.float32, name ="W") ;
b = tf.Variable(npr.uniform(-0.01,0.01, [1,10]),dtype=tf.float32, name ="b") ;


sess.run(tf.global_variables_initializer()) ;
l1 = tf.nn.relu(tf.matmul(data_placeholder, Wh1) + bh1) ;
print(l1)

l2 = tf.nn.relu(tf.matmul(l1, Wh2) + bh2) ;
print(l2)

l3 = tf.nn.relu(tf.matmul(l2, Wh3) + bh3) ;
print(l2)

logits = tf.matmul(l3, W)+b;
print(logits)

lossBySample = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_placeholder) ;
print(lossBySample) ;

loss = tf.reduce_mean(lossBySample) ;

# classification accuracy
nrCorrect = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(logits,axis=1), tf.argmax(label_placeholder,axis=1)), tf.float32)) ;

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.2) ;  # 0.0000 #0.00002
update = optimizer.minimize(loss) ;

iteration = 0 ;

for iteration in range(0,10):

  sess.run(update, feed_dict = fd) ;
  correct, lossVal,_W = sess.run([nrCorrect, loss,W], feed_dict = fd) ;
  print("epoch ", iteration, "acc=", float(correct), "loss=", lossVal, "wmM=", _W.min(), _W.max()) ;

testout = sess.run(logits, feed_dict = {data_placeholder : testd}) ;

testit = 0 ;

f,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3) ;
f.canvas.mpl_connect('button_press_event', test_cb)
plt.show();
ax = f.gca() ;









