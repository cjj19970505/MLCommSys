#使用两层LSTM训练而成

import io
import numpy as np
from numpy import float32
import tensorflow as tf
import matplotlib.pyplot as plt
from Util import FromFileGenerator
import os

class RemoveLastGenerator(FromFileGenerator):
    removeCount = 50
    def __init__(self, phFilePath, binFilePath):
        return super().__init__(phFilePath, binFilePath)
    def PreProcessPerData(self, dataTuple):
        if RemoveLastGenerator.removeCount == 0:
            return dataTuple
        else:
            return (dataTuple[0][:-(RemoveLastGenerator.removeCount*12)], dataTuple[1][:-RemoveLastGenerator.removeCount])

while True:
    try:
        loadSavedFilePath = input("Saved Path(if train from beginning, type Ctrl+Z): ")
    except EOFError:
        initFromFile = False
        print("Train from beginning")
        break
    else:
        if os.path.isfile(loadSavedFilePath):
            initFromFile = True
            print("Train from saved file")
            break

trainDataGenerator = RemoveLastGenerator('.\\Dataset\\exp1\\divided\\train_ph.txt', '.\\Dataset\\exp1\\divided\\train_bin.txt')
testDataGenerator = RemoveLastGenerator('.\\Dataset\\exp1\\divided\\test_ph.txt', '.\\Dataset\\exp1\\divided\\test_bin.txt')
displayDataGenerator = RemoveLastGenerator('.\\Dataset\\exp1\\divided\\disp_ph.txt', '.\\Dataset\\exp1\\divided\\disp_bin.txt')
train_dataset = tf.data.Dataset.from_generator(generator=trainDataGenerator.GetNext, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
test_dataset = tf.data.Dataset.from_generator(generator=testDataGenerator.GetNext, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
display_dataset = tf.data.Dataset.from_generator(generator=displayDataGenerator.GetNext, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
out_sequence_length = 100 - RemoveLastGenerator.removeCount
with tf.variable_scope('TrainingData'):
    trainBatchSize = 100
    trainDataSet = train_dataset.repeat().batch(trainBatchSize)
    trainIterator = trainDataSet.make_one_shot_iterator()
    xTrainBatch, yTrainBatch = trainIterator.get_next()
with tf.variable_scope('TestData'):
    testBatchSize = 196
    testDataSet = test_dataset.repeat().batch(testBatchSize)
    testIterator = testDataSet.make_one_shot_iterator()
    xTestBatch, yTestBatch = testIterator.get_next()
with tf.variable_scope('displayData'):
    displayBatchSize = 1
    displayDataSet = display_dataset.repeat().batch(displayBatchSize)
    displayIterator = displayDataSet.make_one_shot_iterator()
    xDisplayBatch, yDisplayBatch = displayIterator.get_next()

#build model
with tf.variable_scope('TrainingModelInput'):
    xInputTrainBatch = tf.reshape(tensor=xTrainBatch, shape=[trainBatchSize, -1, 12], name='xInputTrainBatch')
    yInputTrainBatch = tf.cast(x=tf.one_hot(indices=yTrainBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputTrainBatch')

with tf.variable_scope('TestModelInput'):
    xInputTestBatch = tf.reshape(tensor=xTestBatch, shape=[testBatchSize, -1, 12], name='xInputTestBatch')
    yInputTestBatch = tf.cast(x=tf.one_hot(indices=yTestBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputTestBatch')
with tf.variable_scope('DisplayModelInput'):
    xInputDisplayBatch = tf.reshape(tensor=xDisplayBatch, shape=[displayBatchSize, -1, 12], name='xInputDisplayBatch')
    yInputDisplayBatch = tf.cast(x=tf.one_hot(indices=yDisplayBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputDisplayBatch')
with tf.variable_scope('Model'):
    with tf.variable_scope('infer'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, None, 12], name='ModelInputX')
        lstm_cell_num = 20
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_cell_num, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_cell_num, forget_bias=1.0)
        rnnout_layer1,_,_ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, tf.unstack(x, out_sequence_length, 1), dtype=tf.float32, scope='rnn_layer1')
        lstm_fw_cell_layer2 = tf.contrib.rnn.BasicLSTMCell(lstm_cell_num, forget_bias=1.0)
        lstm_bw_cell_layer2 = tf.contrib.rnn.BasicLSTMCell(lstm_cell_num, forget_bias=1.0)
        rnnout_layer2,_,_ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_layer2, lstm_bw_cell_layer2, rnnout_layer1, dtype=tf.float32, scope='rnn_layer2')
        rnnout = tf.stack(rnnout_layer2, 1)
        modelOut = tf.contrib.layers.fully_connected(inputs=rnnout, num_outputs=2, activation_fn=None, scope='LogitsOut')
    with tf.variable_scope('train'):
        if initFromFile:
            global_step = tf.get_variable(name='GlobalStep', shape=tf.TensorShape([]), dtype=tf.int32, trainable=False)
        else:
            global_step = tf.get_variable(name='GlobalStep', shape=tf.TensorShape([]), dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='TrainInputY')
        #loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.reshape(tensor=y_, shape=[-1,2]), logits=tf.reshape(tensor=modelOut, shape=[-1,2]), scope='loss')
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=modelOut, scope='loss')
    with tf.variable_scope('output'):
        softmaxOut = tf.nn.softmax(modelOut, axis=-1)
        predictions = tf.argmax(modelOut, axis=-1)
    with tf.variable_scope('evaluate'):
        modelOutPredict = tf.argmax(modelOut, axis=-1)
        evalSetOutput = tf.argmax(y_, axis=-1)
        eval_temp = tf.equal(modelOutPredict, evalSetOutput)
        with tf.variable_scope('Precision_Train'):
            precision_train = tf.reduce_sum(tf.cast(eval_temp, tf.float32)) / tf.reduce_sum(tf.cast(tf.ones_like(eval_temp), tf.float32))
        with tf.variable_scope('Precision_Test'):
            precision_test = tf.reduce_sum(tf.cast(eval_temp, tf.float32)) / tf.reduce_sum(tf.cast(tf.ones_like(eval_temp), tf.float32))
        with tf.variable_scope('Precision_NoRecord'):
            precision_norecord = tf.reduce_sum(tf.cast(eval_temp, tf.float32)) / tf.reduce_sum(tf.cast(tf.ones_like(eval_temp), tf.float32))

    with tf.variable_scope('PositionError'):
        pe_tk = tf.unstack(softmaxOut, axis=-1)
        pe_tk = tf.stack([pe_tk[1], pe_tk[0]], axis=-1)
        pe_tk = tf.reduce_sum(pe_tk * y_, axis=-1)
        positionError = tf.reduce_sum(pe_tk, axis=0)
        
    with tf.variable_scope('summary'):
        tf.summary.scalar(name='loss', tensor=loss)
        tf.summary.scalar(name='TrainSetPrecision', tensor=precision_train)
        tf.summary.scalar(name='TestSetPrecision', tensor=precision_test)
    
    
    train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, optimizer="Adagrad", learning_rate=0.1)
    #train_op2 = tf.contrib.layers.optimize_loss(loss=loss2, global_step=global_step, optimizer="Adagrad", learning_rate=0.1)
    

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir=".\\tflog", graph=tf.get_default_graph())
saver = tf.train.Saver()
stopTrain = False
display = False
displayPositionError = False
printInfoInterval = 1000
with tf.Session() as sess:
    print("Session Started")
    if initFromFile:
        saver.restore(sess, loadSavedFilePath)
    else:
        sess.run(tf.global_variables_initializer())
    #sess.run(xTrainIterator.get_next())
    print("Variables Inited")
    while not stopTrain:
        xInput, yInput = sess.run([xInputTrainBatch, yInputTrainBatch])
        train_feed = {x:xInput, y_:yInput}
        summary,_,gs = sess.run([merged, train_op, global_step], feed_dict=train_feed)
        train_writer.add_summary(summary, gs)
        if gs % printInfoInterval == 0:
            l, p = sess.run([loss, precision_train], feed_dict=train_feed)
            xTestFeed, yTestFeed = sess.run([xInputTestBatch, yInputTestBatch])
            test_feed = {x:xTestFeed, y_:yTestFeed}
            testPrecision = sess.run(precision_test, feed_dict=test_feed)
            print("GlobalStep:"+str(gs), "loss:"+str(l), "Prec:"+str(p), "TestPrec:"+str(testPrecision))
        if displayPositionError:
            displayPositionError = False
            xTestFeed, yTestFeed = sess.run([xInputTestBatch, yInputTestBatch])
            test_feed = {x:xTestFeed, y_:yTestFeed}
            posErr = sess.run(positionError, feed_dict=test_feed)
            figure = plt.figure("PositionErrorInTestBatch")
            plt.bar(range(len(posErr)), posErr)
            plt.show()
        while display:
            dispX, dispY = sess.run([xInputDisplayBatch, yInputDisplayBatch])
            disp_feed = {x:dispX, y_:dispY}
            dispInferY, norecordprec = sess.run([predictions, precision_norecord], feed_dict=disp_feed)
            print("DispPrec:"+str(norecordprec))
            phs = dispX.reshape([displayBatchSize,-1])
            bins = np.argmax(dispY, axis=2)
            inferBins = dispInferY
            t = range(phs.shape[1])
            for i in range(phs.shape[0]):
                figure = plt.figure()
                plt.plot(t, phs[i])
                pltRange = np.max(phs[i])-np.min(phs[i])
                pltBase = np.min(phs[i])
                plt.plot(t, pltRange/2*np.repeat(bins[i], 12)+pltBase)
                plt.plot(t, pltRange/2*np.repeat(inferBins[i], 12)+pltBase+pltRange/2)
            plt.show()
            
    save_path = saver.save(sess, ".\\tfsaved\\model_2_blstm_remove_last_saved_gs"+str(gs)+"\\saved.ckpt")
    print("Model saved in path: %s" % save_path)
train_writer.close()