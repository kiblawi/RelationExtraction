import tensorflow as tf
import numpy as np
import os
from random import shuffle, seed
from sklearn import metrics

seed(10)
tf.set_random_seed(10)
tf.contrib.summary

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def parse(serialized_example):
    """
    parses tfrecord file and reads it into memory
    :param serialized_example: tfrecord parse line
    :return: features and labels of tfrecord
    """

    features = tf.parse_single_example(serialized_example,
                                       features ={"x": tf.FixedLenFeature([], tf.string, default_value=""),
                                                  "y": tf.FixedLenFeature([], tf.string, default_value="")})

    feat = tf.decode_raw(features['x'], tf.int8)
    label = tf.decode_raw(features['y'], tf.int8)

    feat = tf.cast(feat,dtype=tf.float32)
    label = tf.cast(label,dtype=tf.float32)

    return feat, label


def feed_forward(input_tensor, num_hidden_layers, weights, biases,keep_prob):
    """
    feed forward component of neural network
    :param input_tensor: input tensor
    :param num_hidden_layers: number of hidden layers
    :param weights: dictionary of weights
    :param biases: dictionary of biases for network
    :param keep_prob: keep probability for dropout
    :return: value of neural network
    """

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

    if num_hidden_layers != 0:
        out_layer_multiplication = tf.matmul(dropout[num_hidden_layers-1],weights['out'],name='out_layer_mult')
    else:
        out_layer_multiplication = tf.matmul(input_tensor,weights['out'],name = 'out_layer_mult')
    out_layer_bias_addition = tf.add(out_layer_multiplication,biases['out'],name='out_layer_add')

    return out_layer_bias_addition


def feed_forward_train(train_dataset_files, hidden_array, model_dir, num_features, key_order, test_dataset_files=None):
    """
    trains feed forward neural network
    :param train_dataset_files: list dataset files (.tfrecord)
    :param hidden_array: array of hidden nodes
    :param model_dir: directory of where to save model once trained
    :param num_features: number of features in model
    :param key_order: order of relations
    :param test_dataset_files: list of dataset files (.tfrecord) optional
    :return: path of trained model
    """

    # count number of instances in tfrecord files
    training_instances_count = 0
    num_positive_instances = 0
    for fn in train_dataset_files:
        for record in tf.python_io.tf_record_iterator(fn):
            training_instances_count += 1
            result = tf.train.Example.FromString(record)
            if result.features.feature['y'].bytes_list.value!=['\x00']:
                num_positive_instances+=1
    print("training count: ",training_instances_count)
    print("training positives: ",num_positive_instances)
    print("training number_of_features: ", num_features)

    # resets the default graph
    tf.reset_default_graph()

    # network parameters
    num_labels = len(key_order)
    num_epochs = 250
    batch_size = 256
    num_hidden_layers = len(hidden_array)

    # build training dataset
    dataset = tf.data.TFRecordDataset(train_dataset_files)
    dataset = dataset.map(parse, num_parallel_calls=64).prefetch(batch_size * 100)
    #dataset = dataset.repeat(num_epochs).prefetch(batch_size * 100)
    dataset = dataset.shuffle(batch_size * 50).prefetch(buffer_size=batch_size * 100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)

    # build training dataset
    training_accuracy_dataset = tf.data.TFRecordDataset(train_dataset_files)
    training_accuracy_dataset = training_accuracy_dataset.map(parse, num_parallel_calls=64).prefetch(batch_size * 100)
    training_accuracy_dataset = training_accuracy_dataset.batch(batch_size)  # batch size
    training_accuracy_dataset = training_accuracy_dataset.prefetch(5)

    # build iterator
    iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)
    batch_features, batch_labels = iterator.get_next()

    # builds training set iterator
    train_iter = dataset.make_initializable_iterator()
    train_accuracy_iter = training_accuracy_dataset.make_initializable_iterator()

    # get test dataset information and build test dataset
    if test_dataset_files is not None:
        test_instances_count = 0
        num_positive_test_instances = 0
        for fn in test_dataset_files:
            for record in tf.python_io.tf_record_iterator(fn):
                test_instances_count += 1
                result = tf.train.Example.FromString(record)
                if result.features.feature['y'].bytes_list.value != ['\x00']:
                    num_positive_test_instances += 1
        print("test count: ", test_instances_count)
        print("test positives: ", num_positive_test_instances)

        test_dataset = tf.data.TFRecordDataset(test_dataset_files)
        test_dataset = test_dataset.map(parse)
        test_dataset = test_dataset.batch(1024)
        test_iter = test_dataset.make_initializable_iterator()

    # keep probability
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # build weight matrices for neural network layers
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

    # feed forward component of network
    yhat = feed_forward(batch_features, num_hidden_layers, weights, biases, keep_prob)
    prob_yhat = tf.nn.sigmoid(yhat, name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5, name='class_predict')

    global_step = tf.Variable(0, name="global_step")
    #calculate cost and update network via backpropogation with gradientdescent
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=yhat))
    tf.summary.scalar('cost', cost)
    updates = tf.train.AdamOptimizer.minimize(cost)

    correct_prediction = tf.equal(tf.round(prob_yhat), tf.round(batch_labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    # Run stochastic gradient descent
    save_path = None
    merged = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(model_dir + '/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(model_dir + '/test')

        for epoch in num_epochs:
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer)


            while True:
                try:
                    u, tl = sess.run([updates, cost], feed_dict={iterator_handle: train_handle, keep_prob: 0.5})
                except tf.errors.OutOfRangeError:
                    break

            train_accuracy_handle = sess.run(train_accuracy_iter.string_handle())
            sess.run(train_accuracy_iter.initializer)

            total_predicted_prob = np.array([])
            total_labels = np.array([])
            step = 0
            total_loss_value = 0
            total_accuracy_value = 0

            while True:
                try:
                    tl_val,ta_val, predicted_class, b_labels = sess.run([cost,accuracy, class_yhat, batch_labels],
                                                                  feed_dict={
                                                                      iterator_handle: train_accuracy_handle,
                                                                      keep_prob: 1.0})
                    total_predicted_prob = np.append(total_predicted_prob, predicted_class)
                    total_labels = np.append(total_labels, b_labels)
                    total_loss_value += tl_val
                    total_accuracy_value += ta_val

                except tf.errors.OutOfRangeError:
                    break

            total_predicted_prob = total_predicted_prob.reshape((training_instances_count, num_labels))
            total_labels = total_labels.reshape((training_instances_count, num_labels))

            total_accuracy_value = total_accuracy_value / step
            total_loss_value = total_loss_value / step
            acc_summary = tf.Summary()
            loss_summary = tf.Summary()
            acc_summary.value.add(tag='Accuracy', simple_value=total_accuracy_value)
            loss_summary.value.add(tag='Loss', simple_value=total_loss_value)
            train_writer.add_summary(acc_summary, epoch)
            train_writer.add_summary(loss_summary, epoch)
            train_writer.flush()

            for l in range(len(key_order)):
                column_l = total_predicted_prob[:, l]
                column_true = total_labels[:, l]
                label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                print("Epoch = %d,Label = %s: %.2f%% "
                      % (epoch, key_order[l], 100. * label_accuracy))

            if test_dataset_files is not None:
                test_handle = sess.run(test_iter.string_handle())
                sess.run(test_iter.initializer)
                test_y_predict_total = np.array([])
                test_y_label_total = np.array([])
                test_loss_value = 0
                test_accuracy_value = 0
                while True:
                    try:
                        test_loss, test_accuracy, batch_test_predict, batch_test_labels = sess.run(
                            [cost, accuracy, class_yhat, batch_labels], feed_dict={
                                iterator_handle: test_handle, keep_prob: 1.0})
                        test_y_predict_total = np.append(test_y_predict_total, batch_test_predict)
                        test_y_label_total = np.append(test_y_label_total, batch_test_labels)
                        test_loss_value += test_loss
                        test_accuracy_value += test_accuracy

                    except tf.errors.OutOfRangeError:
                        break

                test_y_predict_total = test_y_predict_total.reshape((test_instances_count, 1))
                test_y_label_total = test_y_label_total.reshape((test_instances_count, 1))

                test_accuracy_value = test_accuracy_value / step
                test_loss_value = test_loss_value / step
                test_acc_summary = tf.Summary()
                test_loss_summary = tf.Summary()
                test_acc_summary.value.add(tag='Accuracy', simple_value=test_accuracy_value)
                test_loss_summary.value.add(tag='Loss', simple_value=test_loss_value)
                test_writer.add_summary(test_acc_summary, epoch)
                test_writer.add_summary(test_loss_summary, epoch)
                test_writer.flush()

                for l in range(len(key_order)):
                    column_l = test_y_predict_total[:, l]
                    column_true = test_y_label_total[:, l]
                    label_accuracy = metrics.f1_score(y_true=column_true, y_pred=column_l)
                    print("Epoch = %d,Test Label = %s: %.2f%% "
                          % (epoch, key_order[l], 100. * label_accuracy))



        save_path = saver.save(sess, model_dir)

    return save_path


def neural_network_test_tfrecord(total_dataset_files, model_file):
    """

    :param total_dataset_files: list of tfrecord files for testing
    :param model_file: trained model file path
    :return: predicted probabilities and labels
    """
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
    dataset = dataset.batch(1000)

    total_predicted_prob = np.array([])

    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
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

    print(total_predicted_prob)
    return total_predicted_prob, labels

def neural_network_test(features, labels, model_file):
    """
    test neural network if features fit into memory
    :param features: array of features
    :param labels: array of labels
    :param model_file: path of model file
    :return: predicted probabilities and labels
    """

    print(features.shape)
    print(labels.shape)
    features_placeholder = tf.placeholder(features.dtype, features.shape, name='test_features')
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='test_labels')
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.batch(1024)
    total_predicted_prob = np.array([])
    total_labels = np.array([])

    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess,model_file)
        graph = tf.get_default_graph()
        tensor_names = [t.name for op in graph.get_operations() for t in op.values()]
        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={features_placeholder: features,
                                                   labels_placeholder: labels})
        batch_features_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        while True:
            try:
                predicted_val,batch_features,batch_labels= sess.run([predict_prob,batch_features_tensor,batch_labels_tensor],feed_dict={iterator_handle: new_handle,keep_prob_tensor:1.0})
                total_labels = np.append(total_labels,batch_labels)
                total_predicted_prob = np.append(total_predicted_prob,predicted_val)
            except tf.errors.OutOfRangeError:
                break

    total_predicted_prob = total_predicted_prob.reshape(labels.shape)
    total_labels = total_labels.reshape(labels.shape)
    return total_predicted_prob, total_labels


def neural_network_predict(predict_features,model_file):
    """
    WORK IN PROGRESS!
    :param predict_features:
    :param model_file:
    :return:
    """
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess,model_file)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('output:0')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')

        predicted_val,predict_class = sess.run([predict_prob,predict_tensor],feed_dict={input_tensor:predict_features,keep_prob_tensor:1.0})

    return predicted_val