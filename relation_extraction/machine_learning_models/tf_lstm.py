import tensorflow as tf
import numpy as np
import os
from random import shuffle, seed
from sklearn import metrics

seed(10)
tf.set_random_seed(10)
tf.contrib.summary

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_bin_vec(fname):
    """
    Loads word2vec embeddings from .bin file
    :param fname: filename
    :return: list of words, list of word embedding arrays, dictionary of word to index
    """
    word_vecs = []
    words = []
    word_dict = {}
    index = 0
    with open(fname,"rb") as f:
        header = f.readline()
        vocab_size,layer_size = map(int,header.split())
        binary_len = np.dtype('float32').itemsize*layer_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs.append(np.fromstring(f.read(binary_len), dtype='float32'))
            words.append(word)
            word_dict[word] = index
            index+=1
    words.append('UNKNOWN_WORD')
    words.append('PADDING_WORD')
    word_dict['UNKNOWN_WORD'] = index
    word_dict['PADDING_WORD'] = index+1
    last_vector = word_vecs[-1]
    word_vecs.append(np.random.rand(last_vector.shape[0]))
    word_vecs.append(np.zeros(last_vector.shape, dtype='float32'))
    print('finished loading embeddings')
    return words, word_vecs, word_dict

def parse(serialized_example):
    """
    parses tfrecord file and reads it into memory
    :param serialized_example: tfrecord parse line
    """

    features = tf.parse_single_example(serialized_example,
                                       features ={"dep_path_list": tf.FixedLenFeature([], tf.string, default_value=""),
                                                  "dep_word_feat": tf.FixedLenFeature([], tf.string, default_value=""),
                                                  "dep_path_length": tf.FixedLenFeature([], tf.string, default_value=""),
                                                  "dep_word_length": tf.FixedLenFeature([], tf.string, default_value=""),
                                                  "y": tf.FixedLenFeature([], tf.string, default_value="")})

    dep_path_list = tf.decode_raw(features['dep_path_list'], tf.int32)
    dep_word_feat = tf.decode_raw(features['dep_word_feat'],tf.int32)
    dep_path_length = tf.decode_raw(features['dep_path_length'],tf.int32)
    dep_word_length = tf.decode_raw(features['dep_word_length'],tf.int32)
    label = tf.decode_raw(features['y'], tf.int32)

    dep_path_length = tf.reshape(dep_path_length, []) # converts array back to scalar
    dep_word_length = tf.reshape(dep_word_length, []) # converts array back to scalar
    label = tf.cast(label,dtype=tf.float32)

    return dep_path_list,dep_word_feat,dep_path_length,dep_word_length, label

def lstm_train(train_dataset_files, num_dep_types,num_path_words, model_dir, key_order,test_dataset_files=None,word2vec_embeddings = None):
    """
    Trains LSTM model with word embeddings
    :param train_dataset_files: list of training dataset (.tfrecord) files
    :param num_dep_types: number of dep types
    :param num_path_words: number of dep path words
    :param model_dir: directory where model gets saved
    :param key_order: key order of relations
    :param test_dataset_files: list of testing datset (.tfrecord) files (optional)
    :param word2vec_embeddings: word2vec embedding dictionary
    :return: return saved path
    """
    # training set statistics
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

    # network parameters
    lambda_l2 = 0.00001
    word_embedding_dimension = 200
    word_state_size = 200
    dep_embedding_dimension = 50
    dep_state_size = 50
    num_labels = len(key_order)
    num_epochs = 250
    batch_size = 128

    # build training dataset
    dataset = tf.data.TFRecordDataset(train_dataset_files)
    dataset = dataset.map(parse,num_parallel_calls=64).prefetch(batch_size*100)
    #dataset = dataset.repeat(num_epochs).prefetch(batch_size * 100)
    dataset = dataset.shuffle(batch_size * 50).prefetch(buffer_size=batch_size * 100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)

    # build training dataset
    training_accuracy_dataset = tf.data.TFRecordDataset(train_dataset_files)
    # dataset = dataset.shuffle(10000)
    training_accuracy_dataset = training_accuracy_dataset.map(parse, num_parallel_calls=64).prefetch(batch_size * 100)
    training_accuracy_dataset = training_accuracy_dataset.batch(batch_size)  # batch size
    training_accuracy_dataset = training_accuracy_dataset.prefetch(5)

    # build iterator
    iterator_handle = tf.placeholder(tf.string, shape=[],name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(
        iterator_handle,
        dataset.output_types,
        dataset.output_shapes)
    batch_dependency_ids, batch_word_ids, batch_dependency_type_length, batch_dep_word_length, batch_labels = iterator.get_next()

    #intialize training iterator
    train_iter = dataset.make_initializable_iterator()
    train_accuracy_iter = training_accuracy_dataset.make_initializable_iterator()
    #test dataset files
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


    #maximum_length_path = tf.shape(batch_dependency_ids)[1]

    # keep probability for dropout layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.name_scope("dependency_type_embedding"):
        print(num_dep_types)
        W = tf.Variable(tf.concat([tf.random_uniform([num_dep_types-1, dep_embedding_dimension]),
                                   tf.zeros([1,dep_embedding_dimension])],axis=0), name="W")
        word_zeroes = tf.fill([tf.shape(batch_word_ids)[0],tf.shape(batch_word_ids)[1],word_embedding_dimension],0.0)
        embedded_dep = tf.concat([word_zeroes,tf.nn.embedding_lookup(W, batch_dependency_ids)],axis = 2)
        print(embedded_dep.shape)

        dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})


    if word2vec_embeddings is not None:
        with tf.name_scope("dependency_word_embedding"):
            print('bionlp_word_embedding')
            W = tf.Variable(tf.constant(0.0, shape=[num_path_words, word_embedding_dimension]), name="W")
            embedding_placeholder = tf.placeholder(tf.float32, [num_path_words, word_embedding_dimension])
            embedding_init = W.assign(embedding_placeholder)
            dep_zeroes = tf.fill([tf.shape(batch_dependency_ids)[0],tf.shape(batch_dependency_ids)[1], dep_embedding_dimension],0.0)
            embedded_word = tf.concat([tf.nn.embedding_lookup(W, batch_word_ids),dep_zeroes],axis=2)

            word_embedding_saver = tf.train.Saver({"dependency_word_embedding/W": W})


    else:
        with tf.name_scope("dependency_word_embedding"):
            W = tf.Variable(tf.concat([tf.random_uniform([num_path_words - 1, word_embedding_dimension]),
                           tf.zeros([1, word_embedding_dimension])],axis=0), name="W")
            dep_zeroes = tf.fill([tf.shape(batch_dependency_ids)[0],tf.shape(batch_dependency_ids)[1], dep_embedding_dimension],0.0)
            embedded_word = tf.concat([tf.nn.embedding_lookup(W, batch_word_ids), dep_zeroes],axis=2)
            word_embedding_saver = tf.train.Saver({"dependency_word_embedding/W": W})

    with tf.name_scope("word_dropout"):
        embedded_word_drop = tf.nn.dropout(embedded_word, keep_prob)

    concattenated = tf.concat([tf.expand_dims(embedded_word_drop,2),tf.expand_dims(embedded_dep,2)],2)
    total_embedded = tf.reshape(concattenated,[-1,200,word_embedding_dimension+dep_embedding_dimension])

    total_sequence_length = tf.add(batch_dep_word_length,batch_dependency_type_length)

    initial_hidden_state = tf.zeros([tf.shape(batch_dependency_ids)[0],word_state_size+dep_state_size],name="hidden_state")
    initial_cell_state = tf.zeros([tf.shape(batch_dependency_ids)[0], word_state_size+dep_state_size],
                                    name="cell_state")
    init_states = tf.nn.rnn_cell.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    with tf.variable_scope('lstm'):
        cell = tf.contrib.rnn.LSTMBlockFusedCell(word_state_size+dep_state_size)
        state_series, output_state = cell(tf.transpose(total_embedded, [1, 0, 2]), initial_state=init_states,
                                           sequence_length=total_sequence_length)

        state_series_final = output_state[1]

        #state_series = tf.concat([state_series_dep, state_series_word], 1)


    with tf.name_scope("hidden_layer"):
        W = tf.Variable(tf.truncated_normal([word_state_size + dep_state_size, 100], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([100]), name="b")
        y_hidden_layer = tf.matmul(state_series_final, W) + b


    with tf.name_scope("dropout"):
        y_hidden_layer_drop = tf.nn.dropout(y_hidden_layer, keep_prob)


    with tf.name_scope("sigmoid_layer"):
        W = tf.Variable(tf.truncated_normal([100, num_labels], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([num_labels]), name="b")
        logits = tf.matmul(y_hidden_layer_drop, W) + b
    prob_yhat = tf.nn.sigmoid(logits, name='predict_prob')
    class_yhat = tf.to_int32(prob_yhat > 0.5,name='class_predict')
    # calculating loss values
    tv_all = tf.trainable_variables()
    tv_regu = []
    non_reg = ["dependency_word_embedding/W:0", 'dependency_type_embedding/W:0', "global_step:0", 'hidden_layer/b:0',
               'sigmoid_layer/b:0']
    for t in tv_all:
        if t.name not in non_reg:
            if (t.name.find('biases') == -1):
                tv_regu.append(t)

    with tf.name_scope("loss"):
        l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        total_loss = loss + l2_loss

    correct_prediction = tf.equal(tf.round(prob_yhat), tf.round(batch_labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    global_step = tf.Variable(0, name="global_step")

    optimizer = tf.train.AdamOptimizer().minimize(total_loss, global_step=global_step)

    # Run training
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
        if word2vec_embeddings is not None:
            print('using word2vec embeddings')
            sess.run(embedding_init, feed_dict={embedding_placeholder: word2vec_embeddings})

        for epoch in range(num_epochs):
            train_handle = sess.run(train_iter.string_handle())
            sess.run(train_iter.initializer)

            while True:
                try:
                    u, tl = sess.run([optimizer, loss], feed_dict={iterator_handle: train_handle, keep_prob: 0.5})
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
                    step+=1
                    tl_val, ta_val, predicted_class, b_labels = sess.run([loss, accuracy, class_yhat, batch_labels],
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
                test_step = 0
                test_loss_value = 0
                test_accuracy_value = 0
                while True:
                    try:
                        test_step+=1
                        test_loss, test_accuracy, batch_test_predict, batch_test_labels = sess.run(
                            [total_loss, accuracy, class_yhat, batch_labels], feed_dict={
                                iterator_handle: test_handle, keep_prob: 1.0})
                        test_y_predict_total = np.append(test_y_predict_total, batch_test_predict)
                        test_y_label_total = np.append(test_y_label_total, batch_test_labels)
                        test_loss_value += test_loss
                        test_accuracy_value += test_accuracy

                    except tf.errors.OutOfRangeError:
                        break

                test_y_predict_total = test_y_predict_total.reshape((test_instances_count, 1))
                test_y_label_total = test_y_label_total.reshape((test_instances_count, 1))

                test_accuracy_value = test_accuracy_value / test_step
                test_loss_value = test_loss_value / test_step
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

    return save_path

def lstm_test(test_dep_path_list_features,test_dep_word_features,test_dep_type_path_length,test_dep_word_path_length, test_labels,model_file):
    """
    test instances through lstm network
    :param test_features: list of test features
    :param test_labels: list of test labels
    :param model_file: path of trained lstm model
    :return: predicted probabilities and labels
    """


    dependency_ids = tf.placeholder(test_dep_path_list_features.dtype, test_dep_path_list_features.shape,
                                    name="dependency_ids")
    dependency_type_sequence_length = tf.placeholder(test_dep_type_path_length.dtype,
                                                     test_dep_type_path_length.shape,
                                                     name="dependency_type_sequence_length")
    word_ids = tf.placeholder(test_dep_word_features.dtype, test_dep_word_features.shape, name="word_ids")
    dependency_word_sequence_length = tf.placeholder(test_dep_word_path_length.dtype,
                                                     test_dep_word_path_length.shape,
                                                     name="dependency_word_sequence_length")
    output_tensor = tf.placeholder(tf.float32, test_labels.shape, name='output')
    dataset = tf.data.Dataset.from_tensor_slices((dependency_ids, word_ids, dependency_type_sequence_length,
                                                  dependency_word_sequence_length, output_tensor))
    dataset = dataset.batch(1000)

    total_labels = np.array([])
    total_predicted_prob = np.array([])
    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta',clear_devices=True)
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()

        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer, feed_dict={dependency_ids: test_dep_path_list_features,
                                                       word_ids: test_dep_word_features,
                                                       dependency_type_sequence_length: test_dep_type_path_length,
                                                       dependency_word_sequence_length: test_dep_word_path_length,
                                                       output_tensor: test_labels})
        dependency_ids_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        dependency_words_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        dep_type_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:2')
        dep_word_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:3')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:4')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')
        while True:
            try:
                predicted_val, batch_labels = sess.run(
                    [predict_prob, batch_labels_tensor],
                    feed_dict={iterator_handle: new_handle, keep_prob_tensor: 1.0})
                total_labels = np.append(total_labels, batch_labels)
                total_predicted_prob = np.append(total_predicted_prob, predicted_val)
            except tf.errors.OutOfRangeError:
                break

    print(total_predicted_prob.shape)
    total_predicted_prob = total_predicted_prob.reshape(test_labels.shape)
    total_labels = total_labels.reshape(test_labels.shape)

    return total_predicted_prob, total_labels

def lstm_predict(total_dataset_files,model_file):
    dataset = tf.data.TFRecordDataset(total_dataset_files)
    dataset = dataset.map(parse)
    dataset = dataset.batch(1)


    with tf.Session() as sess:
        restored_model = tf.train.import_meta_graph(model_file + '.meta')
        restored_model.restore(sess, model_file)
        graph = tf.get_default_graph()
        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')
        test_iterator = dataset.make_initializable_iterator()
        new_handle = sess.run(test_iterator.string_handle())
        sess.run(test_iterator.initializer)
        dependency_ids_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
        dependency_words_tensor = graph.get_tensor_by_name('IteratorGetNext:1')
        dep_type_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:2')
        dep_word_sequence_length_tensor = graph.get_tensor_by_name('IteratorGetNext:3')
        batch_labels_tensor = graph.get_tensor_by_name('IteratorGetNext:4')
        keep_prob_tensor = graph.get_tensor_by_name('keep_prob:0')
        predict_tensor = graph.get_tensor_by_name('class_predict:0')
        predict_prob = graph.get_tensor_by_name('predict_prob:0')
        total_predicted_prob = np.array([])
        while True:
            try:
                predicted_val = sess.run([predict_prob], feed_dict={iterator_handle: new_handle, keep_prob_tensor: 1.0})
                total_predicted_prob = np.append(total_predicted_prob, predicted_val[0])
            except tf.errors.OutOfRangeError:
                break

    print(total_predicted_prob)
    return total_predicted_prob

