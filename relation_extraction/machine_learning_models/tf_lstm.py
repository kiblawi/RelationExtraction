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
      "dep_path_list": tf.FixedLenFeature([], tf.string, default_value=""),
      "dep_word_feat": tf.FixedLenFeature([], tf.string, default_value=""),
      "dep_path_length": tf.FixedLenFeature([], tf.string, default_value=""),
      "dep_word_length": tf.FixedLenFeature([], tf.string, default_value=""),
      "y": tf.FixedLenFeature([], tf.string, default_value="")
    })

  dep_path_list = tf.decode_raw(features['dep_path_list'], tf.int32)
  dep_word_feat = tf.decode_raw(features['dep_word_feat'],tf.int32)
  dep_path_length = tf.decode_raw(features['dep_path_length'],tf.int32)
  dep_word_length = tf.decode_raw(features['dep_word_length'],tf.int32)
  label = tf.decode_raw(features['y'], tf.int32)

  #dep_path_list = tf.cast(dep_path_list,dtype=tf.float32)
  #dep_word_feat = tf.cast(dep_word_feat, dtype=tf.float32)
  dep_path_length = tf.reshape(dep_path_length, []) #converts array back to scalar
  dep_word_length = tf.reshape(dep_word_length, []) #converts array back to scalar
  label = tf.cast(label,dtype=tf.float32)

  return dep_path_list,dep_word_feat,dep_path_length,dep_word_length, label

def lstm_train(train_dataset_files, num_dep_types,num_path_words, model_dir, key_order,test_dataset_files=None):

    training_instances_count = 0
    num_positive_instances = 0
    for fn in train_dataset_files:
        for record in tf.python_io.tf_record_iterator(fn):
            training_instances_count += 1
            result = tf.train.Example.FromString(record)
            if result.features.feature['y'].bytes_list.value!=['\x00\x00\x00\x00']:
                num_positive_instances+=1
    print("training count: ",training_instances_count)
    print("training positives: ",num_positive_instances)
    tf.reset_default_graph()
    #build dataset
    dataset = tf.data.TFRecordDataset(train_dataset_files)
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
    batch_dependency_ids, batch_word_ids, batch_dependency_type_length, batch_dep_word_length, batch_labels = iterator.get_next()

    train_iter = dataset.make_initializable_iterator()


    if test_dataset_files is not None:
        test_instances_count = 0
        num_positive_test_instances = 0
        for fn in test_dataset_files:
            for record in tf.python_io.tf_record_iterator(fn):
                test_instances_count += 1
                result = tf.train.Example.FromString(record)
                if result.features.feature['y'].bytes_list.value != ['\x00\x00\x00\x00']:
                    num_positive_test_instances += 1
        print("test count: ", test_instances_count)
        print("test positives: ", num_positive_test_instances)

        test_dataset = tf.data.TFRecordDataset(test_dataset_files)
        test_dataset = test_dataset.map(parse)
        test_dataset = test_dataset.batch(1024)
        test_iter = test_dataset.make_initializable_iterator()




    lambda_l2 = 0.00001
    word_embedding_dimension = 100
    word_state_size = 100
    dep_embedding_dimension = 50
    dep_state_size = 50
    num_labels = len(key_order)
    num_epochs = 60
    maximum_length_path = tf.shape(batch_dependency_ids)[1]


    with tf.name_scope("dependency_type_embedding"):
        W = tf.Variable(tf.random_uniform([num_dep_types, dep_embedding_dimension]), name="W")
        embedded_dep = tf.nn.embedding_lookup(W, batch_dependency_ids)
        dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})

    with tf.name_scope("dependency_word_embedding"):
        W = tf.Variable(tf.random_uniform([num_path_words, word_embedding_dimension]), name="W")
        embedded_word = tf.nn.embedding_lookup(W, batch_word_ids)
        word_embedding_saver = tf.train.Saver({"word_embedding/W": W})

    with tf.name_scope("word_dropout"):
        embedded_word_drop = tf.nn.dropout(embedded_word, 0.3)

    dependency_hidden_states = tf.zeros([tf.shape(batch_dependency_ids)[0], dep_state_size], name="dep_hidden_state")
    dependency_cell_states = tf.zeros([tf.shape(batch_dependency_ids)[0], dep_state_size], name="dep_cell_state")
    dependency_init_states = tf.nn.rnn_cell.LSTMStateTuple(dependency_hidden_states, dependency_cell_states)

    word_hidden_state = tf.zeros([tf.shape(batch_word_ids)[0], word_state_size], name='word_hidden_state')
    word_cell_state = tf.zeros([tf.shape(batch_word_ids)[0], word_state_size], name='word_cell_state')
    word_init_state = tf.nn.rnn_cell.LSTMStateTuple(word_hidden_state, word_cell_state)

    with tf.variable_scope("dependency_lstm"):
        cell = tf.contrib.rnn.BasicLSTMCell(dep_state_size)
        state_series, current_state = tf.nn.dynamic_rnn(cell, embedded_dep, sequence_length=batch_dependency_type_length,
                                                        initial_state=dependency_init_states)
        state_series_dep = tf.reduce_max(state_series, axis=1)

    with tf.variable_scope("word_lstm"):
        cell = tf.nn.rnn_cell.BasicLSTMCell(word_state_size)
        state_series, current_state = tf.nn.dynamic_rnn(cell, embedded_word_drop, sequence_length=batch_dep_word_length,
                                                        initial_state=word_init_state)
        state_series_word = tf.reduce_max(state_series, axis=1)

    state_series = tf.concat([state_series_dep, state_series_word], 1)

    with tf.name_scope("hidden_layer"):
        W = tf.Variable(tf.truncated_normal([dep_state_size + word_state_size, 256], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([256]), name="b")
        y_hidden_layer = tf.matmul(state_series, W) + b


    with tf.name_scope("dropout"):
        y_hidden_layer_drop = tf.nn.dropout(y_hidden_layer, 0.3)

    with tf.name_scope("sigmoid_layer"):
        W = tf.Variable(tf.truncated_normal([256, num_labels], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([num_labels]), name="b")
        logits = tf.matmul(y_hidden_layer_drop, W) + b
        prob_yhat = tf.nn.sigmoid(logits, name='predict_prob')

    tv_all = tf.trainable_variables()
    tv_regu = []
    non_reg = ["dependency_word_embedding/W:0", 'dependency_type_embedding/W:0', "global_step:0", 'hidden_layer/b:0',
               'sigmoid_layer/b:0']
    for t in tv_all:
        if t.name not in non_reg:
            if (t.name.find('biases') == -1):
                tv_regu.append(t)

    with tf.name_scope("loss"):
        #l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        total_loss = loss #+ l2_loss

    global_step = tf.Variable(0, name="global_step")

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, global_step=global_step)

    #saver = tf.train.Saver()
    # Run SGD
    save_path = None
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #saver = tf.train.Saver()
        #writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())
        for epoch in range(num_epochs):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer)
            print("epoch: ", epoch)
            while True:
                try:
                    #print(sess.run([y_hidden_layer],feed_dict={iterator_handle:train_handle}))
                    _, loss, step = sess.run([optimizer, total_loss, global_step],
                                             feed_dict={iterator_handle: train_handle})
                    print("Step:", step, "loss:", loss)
                    #print("Step:", step, "loss:", loss)
                    #save_path = saver.save(sess, model_dir)
                except tf.errors.OutOfRangeError:
                    break



def neural_network_test_tfrecord(total_dataset_files,model_file):
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
        iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(iterator.string_handle())
        sess.run(iterator.initializer)
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
