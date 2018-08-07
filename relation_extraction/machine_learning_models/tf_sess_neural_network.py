import tensorflow as tf
import numpy as np
from random import shuffle, seed
from sklearn import metrics

seed(10)
tf.set_random_seed(10)

def parse(serialized_example):
  features = tf.parse_single_example(
    serialized_example,
    features ={
      "x": tf.FixedLenFeature([], tf.string, default_value=""),
      "y": tf.FixedLenFeature([], tf.string, default_value="")
    })

  feat = tf.decode_raw(features['x'], tf.int8)
  label = tf.decode_raw(features['y'], tf.int8)

  feat = tf.cast(feat,dtype=tf.float32)
  label = tf.cast(label,dtype=tf.float32)

  return feat, label

def feed_forward(input_tensor, num_hidden_layers, weights, biases,keep_prob):
    """Performs feed forward portion of neural network training"""
    hidden_mult = {}
    hidden_add = {}
    hidden_act  = {}
    dropout = {}


    for i in range(num_hidden_layers):
        if i == 0:
            hidden_mult[i] = tf.matmul(input_tensor,weights[i],name='hidden_mult'+str(i))
        else:
            hidden_mult[i] = tf.matmul(hidden_act[i-1],weights[i],name='hidden_mult'+str(i))
        hidden_add[i] = tf.add(hidden_mult[i], biases[i],'hidden_add'+str(i))
        hidden_act[i] = tf.nn.relu(hidden_add[i],'hidden_act'+str(i))
        dropout[i] = tf.nn.dropout(hidden_act[i], keep_prob)

    #with tf.name_scope('out_activation'):
    if num_hidden_layers != 0:
        out_layer_multiplication = tf.matmul(dropout[num_hidden_layers-1],weights['out'],name='out_layer_mult')
    else:
        out_layer_multiplication = tf.matmul(input_tensor,weights['out'],name = 'out_layer_mult')
    out_layer_bias_addition = tf.add(out_layer_multiplication,biases['out'],name='out_layer_add')
    #out_layer_activation = out_layer_bias_addition
    #out_layer_activation = tf.identity(out_layer_bias_addition, name='out_layer_activation')

    return out_layer_bias_addition



def neural_network_train_tfrecord(total_dataset_files, hidden_array, model_dir, num_features, key_order):
    c = 0
    positive = 0
    for fn in total_dataset_files:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
            result = tf.train.Example.FromString(record)
            if result.features.feature['y'].bytes_list.value!=['\x00']:
                positive+=1
    print("count: ",c)
    print("positives: ",positive)
    tf.reset_default_graph()
    num_epochs=250
    num_labels = len(key_order)
    print("number_of_features: ",num_features)
    num_hidden_layers = len(hidden_array)
    #build dataset
    dataset = tf.data.TFRecordDataset(total_dataset_files)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(10000)
    #dataset = dataset.repeat(10)
    dataset = dataset.batch(512)

    iterator_handle = tf.placeholder(tf.string, shape=[],name='iterator_handle')
    #tf.add_to_collection('iterator_handle',iterator_handle)

    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)
    train_iter = dataset.make_initializable_iterator()
    training_features, training_labels = iterator.get_next()


    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #keep_prob = tf.constant(0.5)
    weights = {}
    biases = {}
    previous_layer_size = num_features
    for i in range(num_hidden_layers):
        num_hidden_units = hidden_array[i]
        weights[i] = tf.Variable(tf.random_normal([previous_layer_size, num_hidden_units], stddev=0.1),
                                 name='weights' + str(i))
        biases[i] = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.1), name='biases' + str(i))
        previous_layer_size = num_hidden_units
    weights['out'] = tf.Variable(tf.random_normal([previous_layer_size, num_labels], stddev=0.1), name='out_weights')
    biases['out'] = tf.Variable(tf.random_normal([num_labels], stddev=0.1), name='out_bias')

    #with tf.device("/gpu:0"):
    yhat = feed_forward(training_features, num_hidden_layers, weights, biases, keep_prob)
    prob_yhat = tf.nn.sigmoid(yhat, name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5, name='class_predict')

    #with tf.device("/gpu:0"):
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=training_labels, logits=yhat)
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()
    # Run SGD
    save_path=None
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())
        for epoch in range(num_epochs):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer)
            print("epoch: ",epoch)
            while True:
                try:
                    u=sess.run([updates],feed_dict={iterator_handle:train_handle,keep_prob:0.5})
                    save_path = saver.save(sess,model_dir)
                except tf.errors.OutOfRangeError:
                    break

    return save_path

def neural_network_train(train_X,train_y,test_X,test_y,hidden_array,model_dir,key_order):
    num_features = train_X.shape[1]
    num_labels = train_y.shape[1]
    #train_y = np.eye(num_labels)[train_y]
    #test_y = np.eye(num_labels)[test_y]
    #num_labels = 2
    num_hidden_layers = len(hidden_array)

    tf.reset_default_graph()

    #with tf.name_scope('input_features_labels'):
    input_tensor = tf.placeholder(tf.float32, [None, num_features], name='input')
    output_tensor = tf.placeholder(tf.float32, [None, num_labels], name='output')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    #with tf.name_scope('weights'):
    weights = {}
    biases = {}
    previous_layer_size = num_features
    for i in range(num_hidden_layers):
        num_hidden_units = hidden_array[i]
        weights[i] = tf.Variable(tf.random_normal([previous_layer_size, num_hidden_units], stddev=0.1),name='weights' + str(i))
        biases[i] = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.1), name='biases' + str(i))
        previous_layer_size = num_hidden_units
    weights['out'] = tf.Variable(tf.random_normal([previous_layer_size, num_labels], stddev=0.1),name='out_weights')
    biases['out'] = tf.Variable(tf.random_normal([num_labels], stddev=0.1), name='out_bias')

    #with tf.name_scope('biases'):
    #    biases = {}
    #    for i in range(num_hidden_layers):
    #        num_hidden_units = hidden_array[i]
    #        biases[i] = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.1), name='biases' + str(i))
    #    biases['out'] = tf.Variable(tf.random_normal([num_labels], stddev=0.1), name='out_bias')

    # Forward propagation
    yhat = feed_forward(input_tensor, num_hidden_layers, weights, biases, keep_prob)
    prob_yhat = tf.nn.sigmoid(yhat,name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5,name='class_predict')
    #predict = tf.argmax(prob_yhat, axis=1,name='predict_tensor')

    # Backward propagation
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=output_tensor, logits=yhat)
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()
    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())

        values = range(len(train_X))

        max_accuracy = 0
        save_path = None
        for epoch in range(250):
            shuffle(values)
            # Train with each example
            for i in values:
                u = sess.run([updates],
                                   feed_dict={input_tensor: train_X[i: i + 1], output_tensor: train_y[i: i + 1],
                                              keep_prob: 0.5})

            save_path = saver.save(sess, model_dir)

            if test_X is not None and test_y is not None:
                train_y_pred = sess.run(class_yhat,feed_dict={input_tensor: train_X, output_tensor: train_y,keep_prob: 1.0})
                test_y_pred =  sess.run(class_yhat,feed_dict={input_tensor: test_X, output_tensor: test_y,keep_prob: 1.0})
                train_accuracy = metrics.accuracy_score(y_true=train_y,y_pred=train_y_pred)
                test_accuracy = metrics.accuracy_score(y_true=test_y, y_pred=test_y_pred)
                for l in range(len(key_order)):
                    column_l = test_y_pred[:,l]
                    column_true = test_y[:,l]
                    label_accuracy = metrics.accuracy_score(y_true=column_true,y_pred=column_l)

                    print("Epoch = %d,Label = %s: %.2f%%, train accuracy = %.2f%%, test accuracy = %.2f%%"
                        % (epoch + 1, key_order[l],100. * label_accuracy, 100. * train_accuracy, 100. * test_accuracy))

    return save_path

def neural_network_test(test_features,test_labels,model_file):
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess,model_file)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('output:0')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        predicted_val,predict_class = sess.run([predict_prob,predict_tensor],feed_dict={input_tensor:test_features,output_tensor:test_labels,keep_prob_tensor:1.0})
        test_accuracy = metrics.accuracy_score(y_true=test_labels, y_pred=predict_class)
        print(test_accuracy)
    return predicted_val

def neural_network_test_tfrecord(total_dataset_files,model_file):
    print(model_file)
    c = 0
    positive = 0
    labels = []
    for fn in total_dataset_files:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
            result = tf.train.Example.FromString(record)
            if result.features.feature['y'].bytes_list.value != ['\x00']:
                labels.append(1)
                positive += 1
            else:
                labels.append(0)
    labels = np.array(labels)
    print("count: ", c)
    print("positives: ", positive)

    dataset = tf.data.TFRecordDataset(total_dataset_files)
    dataset = dataset.map(parse)
    dataset = dataset.batch(1)

    total_predicted_prob = np.array([])

    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess,model_file)
        graph =tf.get_default_graph()
        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        iterator = dataset.make_one_shot_iterator()
        new_handle = sess.run(iterator.string_handle())
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')
        while True:
            try:
                predicted_val, predict_class = sess.run([predict_prob,predict_tensor],feed_dict={iterator_handle: new_handle,keep_prob_tensor:1.0})
                total_predicted_prob = np.append(total_predicted_prob,predicted_val[0])
            except tf.errors.OutOfRangeError:
                break
        #test_accuracy = metrics.accuracy_score(y_true=test_labels, y_pred=predict_class)

    print(total_predicted_prob)
    return total_predicted_prob, labels

def neural_network_test_large(features,labels,model_file):

    print(features.shape)
    print(labels.shape)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(1024)
    total_predicted_prob = np.array([])
    total_labels = np.array([])
    feature_batch, label_batch = tf.contrib.data.get_single_element(dataset)

    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess,model_file)
        graph =tf.get_default_graph()
        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        iterator = dataset.make_one_shot_iterator()
        new_handle = sess.run(iterator.string_handle())
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        while True:
            try:
                predicted_val, predict_class,f_batch, l_batch= sess.run([predict_prob,predict_tensor,feature_batch,label_batch],feed_dict={iterator_handle: new_handle,keep_prob_tensor:1.0})
                print(f_batch)
                print(l_batch)
                print(predicted_val)
                total_predicted_prob = np.append(total_predicted_prob,predicted_val)
                total_labels = np.append(total_labels,l_batch)
            except tf.errors.OutOfRangeError:
                break
        #test_accuracy = metrics.accuracy_score(y_true=test_labels, y_pred=predict_class)

    print(total_predicted_prob.shape)
    total_predicted_prob = total_predicted_prob.reshape(labels.shape)
    return total_predicted_prob, labels


def neural_network_predict(predict_features,model_file):


    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess,model_file)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('output:0')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        predicted_val,predict_class = sess.run([predict_prob,predict_tensor],feed_dict={input_tensor:predict_features,keep_prob_tensor:1.0})

    return predicted_val