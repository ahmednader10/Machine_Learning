import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)
X_testing= test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

X_scaler  = MinMaxScaler(feature_range=(0,1))
Y_scaler  = MinMaxScaler(feature_range=(0,1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

learning_rate= 0.001
training_epochs=100
display_step=5

number_of_inputs = 9
number_of_outputs = 1

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=layer_1_nodes, initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=layer_2_nodes, initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=layer_3_nodes, initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=number_of_outputs, initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.variable_scope('logging'):
    tf.summary.scalar('current cost', cost)
    tf.summary.histogram('predicted_value', prediction)
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    training_writer = tf.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)

    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # print("training pass: {}".format(epoch))
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost,summary], feed_dict={X:X_scaled_training, Y:Y_scaled_training})
            testing_cost, testing_summary = session.run([cost,summary], feed_dict={X:X_scaled_testing, Y:Y_scaled_testing})

            print(epoch, training_cost, testing_cost)

            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
    print("training is complete")

    final_training_cost = session.run(cost, feed_dict={X:X_scaled_training, Y:Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X:X_scaled_testing, Y:Y_scaled_testing})
    print("final training cost: {}".format(final_training_cost))
    print("final testing cost: {}".format(final_testing_cost))

    save_path = saver.save(session, "logs/trained_model.ckpt")
    print("Model saved to: {}".format(save_path))
    #To restore the saved model:
    #saver.restore(session, "logs/trained_model.ckpt")
    #Also we have to comment the tf.global_variables_initializer line so as
    #not to initialize the global variables again

    #To save a model to be deployed in the cloud:
    # model_builder = tf.saved_model.builder.SavedModelBuilder("exported model")
    #
    # inputs = {
    #     'inputs':tf.saved_model.utils.build_tensor_info(X)
    # }
    #
    # outputs = {
    #     'earnings':tf.saved_model.utils.build_tensor_info(prediction)
    # }
    #
    # signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    #     inputs = inputs,
    #     outputs = outputs,
    #     method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    #
    # )
    #
    # model_builder.add_meta_graph_and_variables(
    #     session,
    #     tags = [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    #     }
    # )
    # model_builder.save()
