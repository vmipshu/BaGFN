# -coding:utf-8-
"""
File Name:BLS
Description:
        Author:Tony
        Date:2020/1/13 9:19
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class BLS(object):
    """ Broad Learning System for BaGFN """

    def __init__(self, Fields_size, embedding_size, NumEnhan, MapNode):

        self.F = Fields_size
        self.k = embedding_size
        self.NumEnhan = NumEnhan
        self.MapNode = MapNode
        self.initializer = xavier_initializer()

    def mapping_node(self, data):
        """
        :param input: (b,Fi,1)
        :param whiten:
        :return: (b,Fi,mappingNode)
        """
        weights = self.weights("mappingNode", 1, 0)
        bias = self.bias("mappingNode", 1, 0)
        out0 = tf.matmul(data, weights) + bias

        input = out0
        result = out0
        for i in range(1, self.MapNode):
            weights = self.weights("mappingNode", 1, i)
            bias = self.bias("mappingNode", 1, i)
            out_temp = tf.matmul(input, weights) + bias
            out_temp = tf.nn.leaky_relu(out_temp)
            input = out_temp
            result = tf.concat([result, out_temp], axis=-1)

        print("mapping_node:", result.shape)

        return result

    def enhance_node(self, data):
        """
        :param data: (B,fi,mapNode)
        :return: (B,fi,enhanceNode)
        """
        weights = self.weights("enhanceNode", 1, 0)
        bias = self.bias("enhanceNode", 1, 0)
        out0 = tf.matmul(data, weights) + bias

        input = out0
        out0 = tf.reduce_mean(out0, axis=-1, keep_dims=True)
        result = out0

        for i in range(1, self.NumEnhan):
            weights = self.weights("enhanceNode", 1, i)
            bias = self.bias("enhanceNode", 1, i)
            out_temp = tf.matmul(input, weights) + bias
            out_temp = tf.nn.leaky_relu(out_temp)
            input = out_temp
            # out_temp=tf.reduce_mean(out_temp,axis=-1,keep_dims=True)
            result = tf.concat([result, out_temp], axis=-1)

        return result

    def weights(self, name, node_size, i):

        if name == "mappingNode":
            w = tf.compat.v1.get_variable(name="mapping_weights_" + str(i),
                                          shape=[node_size, node_size],
                                          initializer=self.initializer
                                          )
        elif name == "enhanceNode":
            if i == 0:
                w = tf.compat.v1.get_variable(name="enahcne_weights_" + str(i),
                                              # shape=[self.MapNode,self.NumEnhan],
                                              shape=[node_size, node_size],
                                              initializer=self.initializer
                                              )
            else:
                w = tf.compat.v1.get_variable(name="enahcne_weights_" + str(i),
                                              # shape=[self.NumEnhan, self.NumEnhan],
                                              shape=[node_size, node_size],
                                              initializer=self.initializer
                                              )
        return w

    def bias(self, name, node_size, i):

        if name == "mappingNode":
            b = tf.compat.v1.get_variable(name="mapping_bias_" + str(i),
                                          shape=[node_size, ],
                                          initializer=self.initializer
                                          )
        elif name == "enhanceNode":
            b = tf.compat.v1.get_variable(name="enahcne_bias_" + str(i),
                                          # shape=[self.NumEnhan,],
                                          shape=[node_size, ],
                                          initializer=self.initializer
                                          )
        return b

    def generate_node(self, input, whiten=False):
        """
        :param input: (b,Fi,1)
        :param whiten:
        :return: (b,Fi,mappingNode)
        """
        if whiten:
            weights = tf.compat.v1.get_variable(name="features_whiten",
                                                shape=[self.MapNode, self.NumEnhan],
                                                initializer=self.initializer
                                                )  #
            bias = tf.compat.v1.get_variable(name="bias_whiten",
                                             shape=[self.NumEnhan, ],
                                             initializer=self.initializer,
                                             )  #
        else:
            weights = tf.compat.v1.get_variable(name="features",
                                                shape=[1, self.MapNode],
                                                initializer=self.initializer
                                                )  #
            bias = tf.compat.v1.get_variable(name="bias",
                                             shape=[self.MapNode, ],
                                             initializer=self.initializer
                                             )  #
        map_out = tf.matmul(input, weights) + bias
        output = tf.nn.leaky_relu(map_out)

        return output

    def pinv_mat(self, data):
        """
        :param data: (B,Fi,k)
        :return: (B,Fi,k)
        """
        r = tf.Variable(0.1, trainable=True)
        E = r * tf.eye(data.get_shape().as_list()[-1])
        ATA = tf.matmul(tf.transpose(data, [0, 2, 1]), data)
        part1 = tf.linalg.inv(ATA + E)
        part2 = tf.transpose(data, [0, 2, 1])
        out = tf.matmul(part1, part2)
        out = tf.transpose(out, [0, 2, 1])

        return out

    def fit(self, data):

        data = tf.reduce_mean(data, axis=-1, keep_dims=True)  # (b,Fi,1)
        # print('data:',data.shape)
        # mappingdata=self.generate_node(data)#(b,Fi,mapNode)
        mappingdata = self.mapping_node(data)

        w = tf.compat.v1.get_variable(name="weights",
                                      shape=[self.MapNode, ],
                                      initializer=self.initializer
                                      )
        enhance_input = mappingdata * w
        # print("enhance_input:",enhance_input.shape)
        # enhancedata=self.generate_node(enhance_input,whiten=True)#(b,Fi,EnhanNode)
        enhance_input = tf.reduce_mean(enhance_input, axis=-1, keep_dims=True)  # (b,Fi,1)
        enhancedata = self.enhance_node(enhance_input)

        inputdata = tf.concat([mappingdata, enhancedata], axis=-1)
        # print("input:",inputdata.shape)
        pesuedoinverse = self.pinv_mat(inputdata)  # (B,Fi,map+NumNode)
        # print("persue:", pesuedoinverse.shape)
        bls_out = tf.reduce_mean(pesuedoinverse, axis=-1, keep_dims=True)
        bls_out = tf.nn.sigmoid(bls_out)

        return bls_out
