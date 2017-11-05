#!encoding=utf8
import tensorflow as tf
import numpy as np

import collections, itertools
import os, math, jieba, random, logging
import codecs

# Step 1: Download the data.
# Read the data into a list of strings.
def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    #读取停用词
    stop_words = []
    with codecs.open('stop_words.txt',"r", "utf8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，得到词典
    sentence_list = []
    line_list = []
    with codecs.open('doupocangqiong.txt',"r", "utf8") as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            while ' ' in line:
                line = line.replace(' ','')
            line = line.strip()
            if len(line)>0: # 如果句子非空
                raw_words = list(jieba.cut(line,cut_all=False))
                raw_words = [w for w in raw_words if w not in stop_words]
                sentence_list.append(raw_words)
                line_list.append(line)
            line=f.readline()
    return sentence_list, line_list

#step 1:读取文件中的内容组成一个列表
sentence_list, line_list = read_data()
print('sentence len ', len(sentence_list))

vocabulary_size = 50000
doc_size = len(sentence_list)

def build_dataset(words, vocabulary_size):
    count = [['UNKNOWN', -1], ]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def build_docs(sentence_list):
    global vocabulary_size
    sentence_ids = []
    words = []
    for i, sentence in enumerate(sentence_list):
        sentence_ids.extend([i] * len(sentence))
        words.extend(sentence)
    word_ids, count, dictionary, reversed_dictionary = build_dataset(words, vocabulary_size)
    return sentence_ids, word_ids, count, dictionary, reversed_dictionary

doc_ids, word_ids, _, dictionary, reversed_dictionary = build_docs(sentence_list)

print 'sentence cnt:', len(doc_ids), ' word cnt:', len(word_ids), 'word_max', max(word_ids)
data_index = 0

def generate_batch_pvdm(doc_ids, word_ids, batch_size, window_size):
    global data_index
    assert batch_size % window_size == 0
    batch = np.ndarray(shape=(batch_size, window_size+1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = window_size + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer_doc = collections.deque(maxlen=span)
    
    buffer.extend(word_ids[data_index: data_index+span])
    buffer_doc.extend(doc_ids[data_index: data_index+span])
    data_index = (data_index + 1) % len(word_ids)
    mask = [1] * span
    mask[-1] = 0
    i = 0
    while i < batch_size:
        if len(set(buffer_doc)) == 1:
            ## one sentence has enough words (window_size+1 words)
            ## batch is [worc_vec1, ... word_vecN, doc_vec]
            #batch[i, :-2] = list(buffer)[:-2]
            #batch[i, -1] = buffer_doc[-1]
            doc_id = buffer_doc[-1]
            batch[i, :] = list(itertools.compress(buffer, mask)) + [doc_id]
            labels[i, 0] = buffer[-1]
            i += 1
        buffer.extend(word_ids[data_index: data_index+span])
        buffer_doc.extend(doc_ids[data_index: data_index+span])
        data_index = (data_index + 1) % len(word_ids)
    return batch, labels



#batch, labels = generate_batch_pvdm(doc_ids, word_ids, 2, 1)
#for i in range(2):
#    print ','.join([str(w_id) + ':' + reversed_dictionary[w_id] for w_id in batch[i, :-2]]), batch[i][-1], '->', labels[i][0], reversed_dictionary[labels[i][0]]

batch_size = 128
embedding_size_word = 128
embedding_size_doc = 128
window_size = 4 # the window [word1, word2 ....] -> target_word
num_sampled = 64 ## neg sample 

INPUT_CONCATE = 1
INPUT_AVERAGE = 2
input_mode = INPUT_AVERAGE

valid_words = [u'萧炎',u'灵魂',u'火焰',u'萧薰儿',u'药老',u'天阶',u"云岚宗",u"乌坦城",u"惊诧", u"少女"]
#valid_words = [u'斗破']
valid_word_examples =[dictionary[li] for li in valid_words]
valid_size = len(valid_word_examples)

valid_doc_examples = [4, 10, 11]
valid_doc_size = len(valid_doc_examples)

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, window_size+1])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_word_ids = tf.constant(valid_word_examples, dtype=tf.int32)
    valid_doc_ids = tf.constant(valid_doc_examples, dtype=tf.int32)
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size_word], -1.0, 1.0))
        embeddings_doc = tf.Variable(tf.random_uniform([doc_size, embedding_size_doc], -1.0, 1.0))
        
        if input_mode == INPUT_AVERAGE:
            input_embedding_size = embedding_size_word + embedding_size_doc
        else:
            input_embedding_size = embedding_size_word * window_size + embedding_size_doc
        
    
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, input_embedding_size], stddev=1.0/math.sqrt(input_embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
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
        embs_doc = tf.nn.embedding_lookup(embeddings_doc, train_inputs[:, window_size])
        embs.append(embs_doc)

        final_emb = tf.concat(embs, 1)
        
        
    loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=final_emb, num_sampled=num_sampled, num_classes=vocabulary_size)
        )

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)
    
    norm_word = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings_word = embeddings / norm_word
    
    norm_doc = tf.sqrt(tf.reduce_mean(tf.square(embeddings_doc), 1, keep_dims=True))
    normalized_embeddings_doc = embeddings_doc / norm_doc

    valid_embeddings_words = tf.nn.embedding_lookup(normalized_embeddings_word, valid_word_ids)
    similarity = tf.matmul(valid_embeddings_words, normalized_embeddings_word, transpose_b=True)
    
    valid_embeddings_docs = tf.nn.embedding_lookup(normalized_embeddings_doc, valid_doc_ids)
    similarity_doc = tf.matmul(valid_embeddings_docs, normalized_embeddings_doc, transpose_b=True)

    ## or similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings, pern=[1,0]))
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

num_steps = 3000001
#num_steps = 10001

with tf.Session(graph=graph) as sess:
    init.run()
    train_writer = tf.summary.FileWriter("cnn_logs/", sess.graph)
    average_loss = 0
    for i in xrange(1, num_steps):
        inputs, labels = generate_batch_pvdm(doc_ids, word_ids, batch_size, window_size)
        #print 'inputs:', inputs
        #print 'labels:', labels
        feed_dict = {train_inputs : inputs, train_labels: labels}

        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if i % 2000 == 0:
            print 'step', i, 'avg_loss', average_loss / 2000
            average_loss = 0
        if i % 10000 == 0:
            sim = similarity.eval()
            for j in xrange(valid_size):
                valid_word = reversed_dictionary[valid_word_examples[j]]
                top_k = 8
                nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                logging.info(log_str)
            sim_doc = similarity_doc.eval()
            for j in xrange(len(valid_doc_examples)):
                doc_near = (-sim_doc[j, :]).argsort()[1]
                print 'source:', line_list[valid_doc_examples[j]]
                print 'near:', line_list[doc_near]

    print 'save model'
    saver.save(sess, "model_pvdm/model.ckpt")
    print 'start to save embeddings ....'
    norm_embeddings_word = normalized_embeddings_word.eval()
    np.savetxt('norm_embeddings_word.txt',norm_embeddings_word)
    norm_embeddings_doc = normalized_embeddings_doc.eval()
    np.savetxt('norm_embedings_sentence.txt', norm_embeddings_doc)
    print  'finished saving'














    

    

