import tensorflow as tf
import numpy as np
import os, json, datetime

from data_load import DataLoader

class SkipThroughSen2vec(object):

    def __init__(self, vocab_size, embedding_dim, num_units, batch_size, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_units = num_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def build_inputs(self):
        with tf.variable_scope("inputs"):
            encode_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encode_inputs")
            encode_inputs_len = tf.placeholder(tf.int32, shape=[None], name="encode_inputs_len")

            pre_decode_inputs = tf.placeholder(tf.int32, shape=[None, None], name="pre_decode_inputs")
            pre_decode_targets = tf.placeholder(tf.int32, shape=[None, None], name="pre_decode_targets")
            pre_decode_inputs_len = tf.placeholder(tf.int32, shape=[None,], name="pre_decode_inputs_len")

            post_decode_inputs = tf.placeholder(tf.int32, shape=[None, None], name="post_decode_inputs")
            post_decode_targets = tf.placeholder(tf.int32, shape=[None, None], name="post_decode_targets")
            post_decode_inputs_len = tf.placeholder(tf.int32, shape=[None,], name="post_decode_inputs_len")
        return encode_inputs, encode_inputs_len, pre_decode_inputs, pre_decode_targets, pre_decode_inputs_len, post_decode_inputs, post_decode_targets, post_decode_inputs_len


    def build_embedding(self, encode_inputs, pre_decode_inputs, post_decode_inputs):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            emb_encode_inputs = tf.nn.embedding_lookup(embedding, encode_inputs)
            emb_pre_decode_inputs = tf.nn.embedding_lookup(embedding, pre_decode_inputs)
            emb_post_decode_inputs = tf.nn.embedding_lookup(embedding, post_decode_inputs)
        return emb_encode_inputs, emb_pre_decode_inputs, emb_post_decode_inputs

    def build_encoder(self, emb_encode_inputs, encode_inputs_len, train=True):
        batch_size = self.batch_size if train else 1
        with tf.variable_scope("encoder"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_units)
            init_state = cell.zero_state(batch_size, tf.float32)
            _, final_state = tf.nn.dynamic_rnn(cell, emb_encode_inputs, initial_state=init_state, sequence_length=encode_inputs_len)
        return init_state, final_state
    
    def softmax_variable(self, num_units, vocab_size, reuse=False):
        with tf.variable_scope("softmax_variable", reuse=reuse):
            w = tf.get_variable("w", shape=[num_units, vocab_size])
            b = tf.get_variable("b", shape=[vocab_size])
        return w, b


    def build_decoder(self, emb_decode_inputs, decode_inputs_len, start_state, scope="decoder", reuse=False):
        with tf.variable_scope(scope):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_units)
            outputs, final_state = tf.nn.dynamic_rnn(cell, emb_decode_inputs, initial_state=start_state, sequence_length=decode_inputs_len)
            x = tf.reshape(outputs, [-1, self.num_units]) #[batch * time_step , num_units]
            w, b = self.softmax_variable(self.num_units, self.vocab_size)
            logits = tf.matmul(x, w) + b
            predictions = tf.nn.softmax(logits, name="predictions")
        return logits, predictions, final_state

    def build_loss(self, logits, targets, scope='loss'):
        with tf.variable_scope(scope):
            target_one_hot = tf.one_hot(targets, self.vocab_size)
            target_shaped = tf.reshape(target_one_hot, [-1, self.vocab_size])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_shaped))
        return loss

    def build_optimizer(self, loss, scope="optimizer"):
        with tf.variable_scope(scope):
            grad_clip = 5
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
            #train_op = tf.train.AdamOptimizer(self.learning_rate)
            train_op = tf.train.AdagradOptimizer(self.learning_rate)
            optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    def build(self):
        encode_inputs, encode_inputs_len, pre_decode_inputs, pre_decode_targets, pre_decode_inputs_len, post_decode_inputs, post_decode_targets, post_decode_inputs_len  = self.build_inputs()
        encode_emb, decode_pre_emb, decode_post_emb = self.build_embedding(encode_inputs, pre_decode_inputs, post_decode_inputs)
        initial_state, final_state = self.build_encoder(encode_emb, encode_inputs_len)

        pre_logits, pre_predictions, pre_final_state = self.build_decoder(decode_pre_emb, pre_decode_inputs_len, final_state, scope="pre_decoder")
        pre_loss = self.build_loss(pre_logits, pre_decode_targets, scope="pre_loss")
        pre_optimizer = self.build_optimizer(pre_loss, "pre_optimizer")
        
        post_logits, post_predictions, post_final_state = self.build_decoder(decode_post_emb, post_decode_inputs_len, final_state, scope="post_decoder")
        post_loss = self.build_loss(post_logits, post_decode_targets, scope="post_loss")
        post_optimizer = self.build_optimizer(post_loss, "post_optimizer")

        inputs = {"initial_state": initial_state, "encoder_inputs": encode_inputs, "encoder_inputs_len": encode_inputs_len,
                "pre_decoder_inputs": pre_decode_inputs, "pre_decoder_inputs_len": pre_decode_inputs_len, "pre_decoder_targets": pre_decode_targets,
                "post_decoder_inputs": post_decode_inputs, "post_decoder_inputs_len": post_decode_inputs_len, "post_decoder_targets": post_decode_targets
                }
        pre_decoder = {"pre_optimizer": pre_optimizer, "pre_loss": pre_loss, "pre_state": pre_final_state}
        post_decoder = {"post_optimizer": post_optimizer, "post_loss": post_loss, "post_state": post_final_state}

        return inputs, pre_decoder, post_decoder


vocab_size = 50000
epoch_num = 1000000
batch_size = 128
data_loader = DataLoader("doupocangqiong.txt", data_file_format="utf8", vocabulary_size=vocab_size, stop_word_file="stop_words.txt", use_jieba=True)
#data_loader = DataLoader("abc_news.txt", data_file_format="ascii", vocabulary_size=vocab_size)
data_loader.load()


def main():
    sen2vec = SkipThroughSen2vec(vocab_size=vocab_size, embedding_dim=64, num_units=64, batch_size=batch_size, learning_rate=1.0)
    inputs, pre_decoder, post_decoder = sen2vec.build()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    new_state = sess.run(inputs['initial_state'])
    step = 0
    while step < epoch_num:
        cur_inputs, cur_inputs_len, pre_inputs, pre_inputs_len, pre_targets, post_inputs, post_inputs_len, post_targets = data_loader.generate_skip_through_batch(batch_size)
        feed = {inputs["initial_state"]: new_state, 
                inputs["encoder_inputs"]: cur_inputs, inputs["encoder_inputs_len"]: cur_inputs_len,
                inputs["pre_decoder_inputs"]: pre_inputs, inputs["pre_decoder_inputs_len"]: pre_inputs_len, inputs["pre_decoder_targets"]: pre_targets,
                inputs["post_decoder_inputs"]: post_inputs, inputs["post_decoder_inputs_len"]: post_inputs_len, inputs["post_decoder_targets"]: post_targets
            }
        ##TODO: why need new_state
        _, pre_loss, _, _, post_loss, new_state = sess.run([
                pre_decoder['pre_optimizer'], pre_decoder['pre_loss'], pre_decoder['pre_state'],
                post_decoder['post_optimizer'], post_decoder['post_loss'], post_decoder['post_state']
                ],
                feed_dict=feed
            )
        if step  % 10 == 0:
            print datetime.datetime.now().strftime('%c'), 'pre_loss:', pre_loss, "post_loss:", post_loss
        step += 1
    #saver.save(sess, "skip_through_models/sen2vec.model")
    sess.close()

if __name__ == "__main__":
    main()


        


