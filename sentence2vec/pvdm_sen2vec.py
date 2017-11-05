#!encoding=utf8
import tensorflow as tf
import numpy as np

import collections, itertools
import os, math, jieba, random, logging
import codecs

from data_load import DataLoader


INPUT_AVERAGE = "avg"
INPUT_CONCAT = "concat"

class Sentence2Vec(object):
    
    def __init__(self, embedding_word_size, embedding_sentence_size, vocabulary_size, sentence_size, window_size=2, input_mode=INPUT_CONCAT,
            batch_size=128, num_sampled=64, loss_type="sampled_softmax_loss"):
        self.embedding_size_word = embedding_word_size
        self.embedding_size_sentence = embedding_sentence_size
        self.vocabulary_size = vocabulary_size
        self.sentence_size = sentence_size
        self.window_size = window_size
        self.input_mode = input_mode
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.loss_type = loss_type
        
        self.build_network()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        train_writer = tf.summary.FileWriter("logs/", self.session.graph)


    def build_network(self):
        batch_size = self.batch_size
        window_size = self.window_size

        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, window_size+1])
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        self.embeddings_word, self.embeddings_sentence, self.embeddings_sentence_online, self.loss, self.loss_online = \
            self.build_loss_layer(self.train_inputs, self.train_labels)
        self.optimizer = self.gen_optimizer(self.loss, METHOD="ADAGRAD", learning_rate=0.1, var_list=None)
        #self.optimizer = self.gen_optimizer(self.loss, METHOD="SGD", learning_rate=0.1, var_list=None)
        #self.optimizer = self.gen_optimizer(self.loss, METHOD="ADAM", learning_rate=0.1, var_list=None)
        #self.optimizer_online = self.gen_optimizer(self.loss_online, METHOD="SGD", learning_rate=0.1, var_list=[self.embeddings_sentence_online])

        self.embeddings_word_normalized = self.normalize_embeddings(self.embeddings_word)
        self.embeddings_sentence_normalized = self.normalize_embeddings(self.embeddings_sentence)
        

        self.valid_word_ids = tf.placeholder(tf.int32, shape=[None,])
        self.similarity_word_layer = self.calc_sim_layer(self.embeddings_word_normalized, self.valid_word_ids)
        self.valid_sentence_ids = tf.placeholder(tf.int32, shape=[None,])
        self.similarity_sentence_layer = self.calc_sim_layer(self.embeddings_sentence_normalized, self.valid_sentence_ids)
        
        
    def build_loss_layer(self, train_inputs, train_labels):
        input_mode = self.input_mode
        batch_size = self.batch_size
        window_size = self.window_size
        vocabulary_size = self.vocabulary_size
        embedding_size_word = self.embedding_size_word
        embedding_size_sentence = self.embedding_size_sentence
        num_sampled = self.num_sampled

        with tf.device('/cpu:0'):
            embeddings = tf.get_variable("embeddings_word", [vocabulary_size, embedding_size_word], initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=tf.float32)
            embeddings_sentence = tf.get_variable("embeddings_sentence", [sentence_size, embedding_size_sentence], initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=tf.float32)
            embeddings_sentence_online = tf.get_variable("embeddings_sentence_online", [1, embedding_size_sentence], initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=tf.float32)
            if input_mode == INPUT_AVERAGE:
                input_embedding_size = embedding_size_word + embedding_size_sentence
            else:
                input_embedding_size = embedding_size_word * window_size + embedding_size_sentence
        
            nce_weights = tf.get_variable("nce_weights", [vocabulary_size, input_embedding_size], 
                    initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(input_embedding_size)), dtype=tf.float32)
            nce_biases = tf.get_variable("nce_biases", [vocabulary_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            
            embs = []
            if input_mode == INPUT_AVERAGE:
                embs_w = tf.zeros([batch_size, embedding_size_word])
                for i in xrange(window_size):
                    embs_w += tf.nn.embedding_lookup(embeddings, train_inputs[:, i])
                embs.append(embs_w)
            else:
                for i in xrange(window_size):
                    embs_w = tf.nn.embedding_lookup(embeddings, train_inputs[:, i])
                    embs.append(embs_w)

            embs_online = []
            embs_online.extend(embs)

            embs_sentence = tf.nn.embedding_lookup(embeddings_sentence, train_inputs[:, window_size])
            embs.append(embs_sentence)
            final_emb = tf.concat(embs, 1)
            
            embs_online.append(tf.nn.embedding_lookup(embeddings_sentence_online, train_inputs[:, window_size]))
            final_emb_online = tf.concat(embs_online, 1)

        if self.loss_type == "nce_loss":
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=final_emb, num_sampled=num_sampled, num_classes=vocabulary_size)
            )
            loss_online = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=final_emb_online, num_sampled=num_sampled, num_classes=vocabulary_size) )
        else:
            loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=final_emb, num_sampled=num_sampled, num_classes=vocabulary_size)
            )
            loss_online = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=final_emb_online, num_sampled=num_sampled, num_classes=vocabulary_size) )

        return embeddings, embeddings_sentence, embeddings_sentence_online, loss, loss_online


    def gen_optimizer(self, loss, METHOD="SGD", learning_rate=0.01, var_list=None):
        if METHOD == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif METHOD == "ADAM":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif METHOD == "ADAGRAD":
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        else:
            raise Exception("not support method!")
        if var_list:
            return optimizer.minimize(loss, var_list=var_list)
        else:
            return optimizer.minimize(loss)

        #optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)
   
    def normalize_embeddings(self, embeddings):
        norm_word = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm_word
        return normalized_embeddings
    
   
    def calc_sim_layer(self, embeddings_normalized, valid_index):
        embeddings_valid = tf.nn.embedding_lookup(embeddings_normalized, valid_index)
        similarity = tf.matmul(embeddings_valid, embeddings_normalized, transpose_b=True)
        return similarity
    
    def calc_sim_word(self, word_ids):
        return self.session.run(self.similarity_word_layer, feed_dict={self.valid_word_ids: word_ids})

    def calc_sim_sentence(self, sentence_ids):
        return self.session.run(self.similarity_sentence_layer, feed_dict={self.valid_sentence_ids: sentence_ids})
    
    
    def train(self, inputs, labels):
        feed_dict = {self.train_inputs : inputs, self.train_labels: labels}
        _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss_val

    def online_train(self, inputs, labels):
        feed_dict = {self.train_inputs : inputs, self.train_labels: labels}
        _, loss_val = self.session.run([self.optimizer_online], feed_dict=feed_dict)
        return loss_val

    def get_online_sentence_vec(self):
        #return self.session.run(self.embeddings_sentence_online)
        return self.embeddings_sentence_online.eval()

    def save_model(self, path):
        self.saver.save(self.session, path)

#valid_words = [u'萧炎',u'灵魂',u'火焰',u'萧薰儿',u'药老',u'天阶',u"云岚宗",u"乌坦城",u"惊诧", u"少女"]
##valid_words = [u'斗破']
#valid_word_examples =[dictionary[li] for li in valid_words]
#valid_size = len(valid_word_examples)
#
#valid_sentence_examples = [4, 10, 11]
#valid_sentence_size = len(valid_sentence_examples)



#data_loader = DataLoader("doupocangqiong.txt", "utf8", vocabulary_size=50000, stop_word_file="stop_words.txt")
VOCA_SIZE = 10000
data_loader = DataLoader("abc_news.txt", "ascii", vocabulary_size=VOCA_SIZE)
data_loader.load()
sentence_size = len(data_loader.line_list)

#batch, labels = data_loader.generate_batch_pvdm(2, 2)
#print 'batch', batch
#print 'labels', labels

embedding_word_size = 64
embedding_sentence_size = 64
batch_size = 128
window_size = 2
num_sampled = 64

valid_sentence_examples = [1, 2]
valid_word_examples = [8, 15]

def train():
    average_loss = 0
    sentence2vec = Sentence2Vec(embedding_word_size, embedding_sentence_size, VOCA_SIZE, sentence_size, 
            window_size=window_size, input_mode="concat", batch_size=batch_size, num_sampled=num_sampled)

    for i in xrange(1, 5000000):
        inputs, labels = data_loader.generate_batch_pvdm(batch_size, window_size)
        loss_val = sentence2vec.train(inputs, labels)
        average_loss += loss_val
        if i % 2000 == 0:
            print 'step', i, 'avg_loss', average_loss / 2000
            average_loss = 0
        if i % 40000 == 0:
            sim = sentence2vec.calc_sim_word(valid_word_examples)
            for j in xrange(len(valid_word_examples)):
                valid_word = data_loader.reversed_dictionary[valid_word_examples[j]]
                top_k = 8
                nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                print 'source word: ' + valid_word
                dest_words = []
                for k in xrange(top_k):
                    close_word = data_loader.reversed_dictionary[nearest[k]]
                    dest_words.append(close_word)
                print 'dest words: ' , ','.join(dest_words)
            print ''
            sim_sentence = sentence2vec.calc_sim_sentence(valid_sentence_examples)
            for j in xrange(len(valid_sentence_examples)):
                sentence_near = (-sim_sentence[j, :]).argsort()[1]
                print 'source sentence: ', data_loader.line_list[valid_sentence_examples[j]]
                print 'near sentence: ', data_loader.line_list[sentence_near]
                print 'sim:', sim_sentence[j, sentence_near]

    #print 'save model'
    #saver.save(sess, "model_pvdm/model.ckpt")
    #print 'start to save embeddings ....'
    #norm_embeddings_word = normalized_embeddings_word.eval()
    #np.savetxt('norm_embeddings_word.txt',norm_embeddings_word)
    #norm_embeddings_sentence = normalized_embeddings_sentence.eval()
    #np.savetxt('norm_embedings_sentence.txt', norm_embeddings_sentence)
    #print  'finished saving'

train()













    

    

