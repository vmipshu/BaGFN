# -coding:utf-8-
"""
File Name:BaGFN
Description:
        Author:Tony
        Date:2020/1/ 
"""
import os
import numpy as np
import tensorflow as tf
from time import time
from sklearn.metrics import roc_auc_score, log_loss

from utils import createlog, create_dir
from models import BLS


class BaGFN(object):
    def __init__(self, args, feature_size, run_cnt):

        self.feature_nums = feature_size  # denote as n, dimension of concatenated features
        self.field_nums = args.field_size  # denote as M, number of total feature fields
        self.embedding_size = args.embedding_size  # denote as d, size of the feature embedding
        self.batch_size = args.batch_size

        self.NumNode = args.field_size
        self.num_gnn = args.num_gnn
        if args.l2_reg > 0:
            self.regu = tf.contrib.layers.l2_regularizer(args.l2_reg)
        else:
            self.regu = None

        self.drop_keep_prob = args.dropout_keep_prob
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type

        self.save_path = os.path.join(args.checkpoint_dir, str(run_cnt))
        self.log_dir = args.log_dir
        create_dir(self.save_path)
        create_dir(self.log_dir)

        self.initializer = None  # xavier_initializer()
        self.logger = createlog(self.log_dir)
        self.random_seed = args.random_seed
        self.best_loss = 1.0
        self.best_AUC = 0.5
        self.BLS = BLS(Fields_size=self.field_nums,
                       embedding_size=self.embedding_size,
                       NumEnhan=self.embedding_size,
                       MapNode=self.embedding_size)

        self._init_graph()

    def _init_graph(self):

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(self.random_seed)

            self.feat_index = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, self.field_nums],
                                                       name="feat_index")  # None * Fi
            self.feat_value = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, self.field_nums],
                                                       name="feat_value")  # None * Fi
            self.label = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, 1], name="label")  # None * 1

            self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, shape=[None], name="dropout_keep_prob")
            self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")
            self.global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)

            # ---------- main part of DGNN-------------------
            emb_x = self.Embedding_layer()
            out = self.main_dgnn(emb_x)
            self.out = tf.nn.sigmoid(out)
            # ---------- Compute the loss ----------
            self.auc_result, self.auc_opt = tf.compat.v1.metrics.auc(self.label, self.out)
            # ---loss---#
            self.loss = tf.compat.v1.losses.log_loss(self.label, self.out)
            # self.loss = self.loss_function()

            # ---optimizer---#
            # self.optimizer = self.optimizer_function()
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

            # ----init-------#
            self.saver = tf.compat.v1.train.Saver(max_to_keep=2)
            init = tf.compat.v1.global_variables_initializer()
            local = tf.compat.v1.local_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.sess.run(local)

    # ---------- Compute the loss ----------
    def loss_function(self):

        self.out = tf.sigmoid(self.out)
        loss = tf.compat.v1.losses.log_loss(self.label, self.out)
        return loss

    # ---------- optimizer function ----------
    def optimizer_function(self):
        if self.optimizer_type == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr) \
                .minimize(self.loss, global_step=self.global_step)

        # add a line
        opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
            optimizer,
            loss_scale='dynamic')

        params = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

        optimizer = opt.apply_gradients(
            zip(clip_gradients, params), global_step=self.global_step)

        return optimizer

    # --------Embedding layer--------#
    def Embedding_layer(self):
        # model
        self.weight_emb = self._initialize_weights("embedding")
        embeddings = tf.nn.embedding_lookup(self.weight_emb,
                                            self.feat_index)  # None * M * d

        feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_nums, 1])
        embeddings = tf.multiply(embeddings, feat_value)  # None * M * d
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        return embeddings

    # -------------main-DGNN-----------#
    def main_dgnn(self, emb_x):
        """
        :param emb_x: (b,Fi,k)
        :return:
        """
        gnn_input = emb_x
        G_A, G_AA = self.Adj(emb_x)  # (b,k,k)
        for i in range(self.num_gnn):
            G_out = self.gnn(gnn_input, G_A, G_AA)  # (b,NumNode,k)
            y_out = self.GRU(G_out, gnn_input)

            gnn_input = y_out + emb_x

        out = self.Node_Interaction(y_out, emb_x)
        return out

    def Node_Interaction(self, GNN_out, x):
        """
        :param GNN_out: (b,NumNode,k)
        :param x: (b,NumNode,k)
        :return: (b,1)
        """
        out1 = tf.matmul(GNN_out
                         , tf.transpose(x, [0, 2, 1]))
        out1 = tf.keras.layers.Dense(self.field_nums,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regu) \
            (tf.reshape(out1, [self.batch_size, -1]))
        out2 = tf.keras.layers.Dense(self.field_nums,
                                     kernel_initializer=self.initializer)(
            tf.reshape(GNN_out + x, [self.batch_size, -1]))
        bls_out = tf.squeeze(self.BLS.fit(GNN_out + x))

        output = out1 * bls_out * out2  # + (1 - bls_out) * out2
        out = tf.keras.layers.Dense(1,
                                    kernel_initializer=self.initializer, ) \
            (tf.reshape(output, [self.batch_size, -1]))

        return out

    def GRU(self, x0, h0):
        """
        GRU cell
        :param input: (b,fi,k)
        :param hidden:(b,fi,k)
        :return: (b,fi,k)
        """
        # x01=self.GRU_cell(x0)#(b,2fi,k)
        # h01=self.GRU_cell(h0)#(b,2fi,k)
        # R_Z=x01+h01
        R_Z = self.GRU_cell(tf.concat([x0, h0], axis=-1))
        r_t, z_t = tf.split(R_Z, num_or_size_splits=2, axis=-1)
        r_t = tf.nn.sigmoid(r_t)  # 重置门 (b,fi,k)
        z_t = tf.nn.sigmoid(z_t)  # 更新门 (b,fi,k)

        h_t = tf.keras.layers.Dense(self.embedding_size,
                                    kernel_initializer=self.initializer,
                                    activation=tf.nn.tanh)(tf.concat([r_t * h0, x0], axis=-1))  # (b,fi,k)
        H_t = (1 - z_t) * h0 + z_t * h_t

        return H_t

    def GRU_cell(self, input):
        """
        :param input: (b,fi,k)
        :return: (6,2fi,k)
        """
        with tf.compat.v1.variable_scope("GRU_cell", reuse=tf.compat.v1.AUTO_REUSE):
            out = tf.keras.layers.Dense(int(self.embedding_size * 2),
                                        kernel_initializer=self.initializer)(input)
            return out

    # ------------gnn------------------#
    def gnn(self, x, G_A, G_AA):
        """
        GNN的实现  (A*(WT*x))*W
        :param x: (b,fi,k)
        :param G_A: (b,fi,fi)
        :return:(b,fi,k)
        """
        with tf.compat.v1.variable_scope("gnn", reuse=tf.compat.v1.AUTO_REUSE):
            # G_A = self.Adj(x)  # (b,k,k)
            # -----------X*W_out---------#
            H0_w = self.weight_in(x)

            # -----GA*(Wout_T*x)----------#
            # --1.---(e_i+e_Nh)e_i-->(I+W)H*H
            A_X_add = tf.matmul(G_A, H0_w) * H0_w  # (b,fi,k)
            # --2.--e_i*ADD(e_Nh)-->W(
            A_X_mul = tf.matmul(G_AA, tf.matmul(G_AA, H0_w) * H0_w)
            A_X = tf.concat([A_X_add * A_X_mul, A_X_mul + A_X_add], axis=-1)
            # A_X = tf.concat([A_X_mul ,A_X_add], axis=-1)

            # ---------GA*X*W_in-------------#
            GNN_out = self.weight_out(A_X)
            GNN = self.attention(GNN_out, H0_w)  # (b,fi,k)
            gnn_out = GNN_out * GNN

        return gnn_out

    def weight_in(self, x):
        """
        W_in
        :param x: (b,Fi,k)
        :return: (b,Fi,k)
        """
        with   tf.compat.v1.variable_scope("gnn", reuse=tf.compat.v1.AUTO_REUSE):
            # -----------X*W_out---------#
            X_W = [tf.matmul(x[:, i, :], self._initialize_weights("W_in", i)) + self._initialize_bias("B_in", i)
                   for i in range(self.NumNode)]
            X_W = tf.concat(X_W, axis=1)
            H0_w = tf.reshape(X_W, [self.batch_size, self.NumNode, self.embedding_size])

        return H0_w

    def weight_out(self, A_X):
        """
        W_out
        :param x: (b,Fi,k)
        :return: (b,Fi,k)
        """
        # ---------GA*X*W_in-------------#
        X_W = [tf.matmul(A_X[:, i, :], self._initialize_weights("W_out", i)) + self._initialize_bias("B_out", i)
               for i in range(self.NumNode)]
        X_W = tf.concat(X_W, axis=1)
        GNN_out = tf.reshape(X_W, [self.batch_size, self.NumNode, self.embedding_size])

        return GNN_out

    def attention(self, G_out, data):
        """
        e_i *e_Niz
        :param G_A: Adj (b,NumNode,NumNode)
        :param X: (b,NumNode,k)
        :return: (b,NumNode,k)
        """
        input = tf.keras.layers.Dense(int(self.embedding_size / 4),
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=self.regu)(
            tf.concat([G_out, data], axis=-1))  # (b,NumNode,k)
        input = tf.keras.layers.Dense(self.embedding_size,
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=self.regu)(input)  # (b,NumNode,k)

        input = input * data
        w = tf.nn.softmax(tf.reduce_mean(input, axis=-1, keepdims=True), axis=-1)
        return w

    def Adj(self, data):
        """
        随机游走获取动态图的邻接矩阵A和w
        input:(b,fi,k)
        return :(b,fi,fi)
        """
        # get A
        a = tf.tile(data, [1, self.NumNode, 1])  # (B,Fi*Fi,k)
        b = tf.tile(data, [1, 1, self.NumNode])  # (B,Fi,k*Fi)

        a = tf.reshape(a, [self.batch_size, self.NumNode, self.NumNode, self.embedding_size])  # (B,Fi,Fi,k)
        b = tf.reshape(b, [self.batch_size, self.NumNode, self.NumNode, self.embedding_size])  # (B,Fi,Fi,K)

        e_i = tf.reshape(a, [self.batch_size, self.NumNode * self.NumNode, self.embedding_size])  # (b,Fi*Fi,k)
        e_j = tf.reshape(b, [self.batch_size, self.NumNode * self.NumNode, self.embedding_size])  # (b,fi*fi,k)

        # --(ei *W* ej)
        e_i = tf.reduce_mean(e_i, -1)
        e_j = tf.reduce_mean(e_j, -1)
        edge_weight = e_i * self._initialize_weights("W_adj", 0) * e_j + self._initialize_bias("B_adj")

        edge_weight = tf.reshape(edge_weight, [self.batch_size, self.NumNode, self.NumNode])  # (b,Fi,Fi)
        E = tf.reshape(tf.tile(tf.eye(self.NumNode), [self.batch_size, 1]),
                       [self.batch_size, self.NumNode, self.NumNode])
        A = tf.ones_like(edge_weight) - E

        A_new1 = edge_weight * A  # (B,NumNode,NumNode) --W*A

        A_new1 = tf.nn.relu(A_new1)
        A_new1 = tf.nn.softmax(A_new1, 1)
        A_new1 = tf.reshape(A_new1, [self.batch_size, self.NumNode, self.NumNode])

        A = tf.ones_like(edge_weight) - E
        A_new2 = edge_weight * A + E  # (B,NumNode,NumNode) --W*A
        A_new2 = tf.nn.relu(A_new2)
        A_new2 = tf.nn.softmax(A_new2, 1)
        A_new2 = tf.reshape(A_new2, [self.batch_size, self.NumNode, self.NumNode])

        return A_new1, A_new2

    def _init_session(self):

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)

    # ------------------weights------------------#
    def _initialize_weights(self, name, i=0):

        # -----embeddings----------#
        hidden_stdv = np.sqrt(1. / (self.embedding_size))
        if name == 'embedding':
            w = tf.compat.v1.get_variable(name="feature_embeddings",
                                          shape=[self.feature_nums, self.embedding_size],
                                          initializer=self.initializer,
                                          )  # feature_size(n) * dk

        # -----------W_adj----------#
        if name == "W_adj":
            w = tf.compat.v1.get_variable(
                name='w/hidden_state_adj',
                shape=[self.NumNode * self.NumNode],
                initializer=self.initializer,
            )
        # -----------w_in----------#
        if name == "W_in":
            w = tf.compat.v1.get_variable(
                name='w/hidden_state_in_' + str(i),
                shape=[self.embedding_size, self.embedding_size],
                initializer=self.initializer,
            )
        # -----------w_out----------#
        if name == "W_out":
            w = tf.compat.v1.get_variable(
                name='w/hidden_state_out_' + str(i),
                shape=[self.embedding_size * 2, self.embedding_size],
                initializer=self.initializer,
            )
        if name == "agg_w":
            w = tf.compat.v1.get_variable(
                name='w/hidden_state_out2_' + str(i),
                shape=[self.NumNode, self.NumNode],
                initializer=self.initializer,
            )

        return w

    def _initialize_bias(self, name, i=0):

        # ----------------B_adj__________#
        if name == "B_adj":
            b = tf.compat.v1.get_variable(
                name='b/hidden_state_adj',
                shape=[self.NumNode * self.NumNode, ],
                initializer=self.initializer,
            )
        # ----------------B_in__________#
        if name == "B_in":
            b = tf.compat.v1.get_variable(
                name='b/hidden_state_in_' + str(i),
                shape=[self.embedding_size, ],
                initializer=self.initializer,
            )
        # ----------------B_out__________#
        if name == "B_out":
            b = tf.compat.v1.get_variable(
                name='b/hidden_state_out_' + str(i),
                shape=[self.embedding_size, ],
                initializer=self.initializer,
            )
        # ----------------B_out__________#
        if name == "B2_out":
            b = tf.compat.v1.get_variable(
                name='b/hidden_state_out2_' + str(i),
                shape=[self.embedding_size, ],
                initializer=self.initializer,
            )

        return b

    def get_batch(self, Xi, Xv, y, batch_size, index):

        start_idx = index * batch_size
        end_idx = (index + 1) * batch_size
        end_idx = min(end_idx, len(y))

        Xi = np.array(Xi[start_idx:end_idx]).astype(np.float)
        Xv = np.array(Xv[start_idx:end_idx]).astype(np.float)
        y = np.array([[y_] for y_ in y[start_idx:end_idx]]).astype(np.float)

        return Xi, Xv, y

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):

        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y, training=True):

        if training:
            feed_dict = {self.feat_index: Xi,
                         self.feat_value: Xv,
                         self.label: y,
                         self.dropout_keep_prob: self.drop_keep_prob,
                         self.train_phase: training}
            step, loss, auc, _, _ = self.sess.run([self.global_step, self.loss,
                                                   self.auc_result, self.auc_opt, self.optimizer], feed_dict=feed_dict)
            return step, loss, auc
        else:
            feed_dict = {self.feat_index: Xi,
                         self.feat_value: Xv,
                         self.label: y,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: training}
            loss, auc, _ = self.sess.run([self.loss, self.auc_result, self.auc_opt], feed_dict=feed_dict)
            return loss, auc

    # ----training-----#
    def fit_once(self, Xi_train, Xv_train, y_train,
                 epoch, file_count,
                 Xi_valid, Xv_valid, y_valid):

        last_step = 0
        t1 = time()
        self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
        total_batch = int(len(y_train) / self.batch_size)
        total_train_loss, total_valid_loss = 0.0, 0.0
        for i in range(total_batch):
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
            step, loss, train_auc = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            last_step = step
            total_train_loss += loss
        train_loss = total_train_loss / total_batch
        train_result = train_auc

        total_batch = int(len(y_valid) / self.batch_size)
        for i in range(total_batch):
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_valid, Xv_valid, y_valid, self.batch_size, i)
            loss, valid_auc = self.fit_on_batch(Xi_batch, Xv_batch, y_batch, False)
            total_valid_loss += loss
        valid_loss = total_valid_loss / total_batch
        valid_result = valid_auc

        # evaluate training and validation datasets
        # train_result, train_loss = self.evaluate(Xi_train, Xv_train, y_train)
        # valid_result, valid_loss = self.evaluate(Xi_valid, Xv_valid, y_valid)

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'), global_step=last_step)

        if valid_result > self.best_AUC:
            self.best_AUC = valid_result

        self.logger.info("[%d-%d] train-result=%.4f, train-logloss=%.4f, "
                         " valid-result=%.4f, valid-logloss=%.4f,Best-AUC=%.4f,Best-Loss=%.4f [%.1f s]"
                         % (epoch, file_count,
                            train_result, train_loss,
                            valid_result, valid_loss,
                            self.best_AUC, self.best_loss, time() - t1))

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * (len(Xi))
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        Xi = Xi[:(int(len(Xi) / self.batch_size) * self.batch_size)]
        Xv = Xv[:(int(len(Xi) / self.batch_size) * self.batch_size)]
        y = y[:(int(len(Xi) / self.batch_size) * self.batch_size)]

        y_pred = self.predict(Xi, Xv)
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

        return roc_auc_score(y, y_pred), log_loss(y, y_pred)

    def restore(self, save_path=None):

        if (save_path == None):
            save_path = self.save_path + '/'
        ckpt = tf.train.get_checkpoint_state(save_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored from %s" % (save_path))
