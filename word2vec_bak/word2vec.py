#!encoding=utf8
import tensorflow as tf
import numpy as np

import collections
import os, math, jieba, random
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
    raw_word_list = []
    with codecs.open('doupocangqiong.txt',"r", "utf8") as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            while ' ' in line:
                line = line.replace(' ','')
            if len(line)>0: # 如果句子非空
                raw_words = list(jieba.cut(line,cut_all=False))
                raw_words = [w for w in raw_words if w not in stop_words]
                raw_word_list.extend(raw_words)
            line=f.readline()
    return raw_word_list

#step 1:读取文件中的内容组成一个列表
words = read_data()
print('Data size', len(words))

vocabulary_size = 50000

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


data, count, dictionary, reversed_dictionary = build_dataset(words, vocabulary_size)

del words

data_index = 0

#def generate_batch(batch_size, num_skip, skip_window):
#    global data_index
#    assert batch_size % num_skip == 0
#    assert num_skip <= 2 * skip_window
#    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#    span = 2 * skip_window + 1
#    buffer = collections.deque(maxlen=span)
#    if data_index + span > len(data):
#        data_index = 0
#    buffer.extend(data[data_index: data_index+span])
#    data_index += span
#    for i in range(batch_size // num_skip):
#        target = skip_window
#        target_to_avoid = [skip_window]
#        for j in range(num_skip):
#            while target in target_to_avoid:
#                target = random.randint(0, span-1)
#            target_to_avoid.append(target)
#            batch[i*num_skip + j] = buffer[skip_window]
#            labels[i*num_skip + j, 0] = buffer[target]
#        if data_index == len(data):
#            for word in data[:span]:
#                buffer.append(word)
#            data_index = span
#        else:
#            buffer.append(data[data_index])
#            data_index += 1
#    data_index = (data_index + len(data) - span) % len(data)
#    return batch, labels

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels



batch, labels = generate_batch(8, 2, 1)
for i in range(8):
    print batch[i], reversed_dictionary[batch[i]], '->', labels[i][0], reversed_dictionary[labels[i][0]]

batch_size = 128
embedding_size = 128
skip_window = 1 # the window [left, right ]
num_skip = 2 # how many pair to generate in the window
num_sampled = 64 ## neg sample 

valid_words = [u'萧炎',u'灵魂',u'火焰',u'萧薰儿',u'药老',u'天阶',u"云岚宗",u"乌坦城",u"惊诧", u"少女"]
valid_examples =[dictionary[li] for li in valid_words]
valid_size = len(valid_words)

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size)
        )

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_examples)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    ## or similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings, pern=[1,0]))
    
    init = tf.global_variables_initializer()

num_steps = 3000001
#num_steps = 10001

with tf.Session(graph=graph) as sess:
    init.run()
    average_loss = 0
    for i in xrange(1, num_steps):
        inputs, labels = generate_batch(batch_size, skip_window, num_skip)
        feed_dict = {train_inputs : inputs, train_labels: labels}

        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if i % 2000 == 0:
            print 'step', i, 'avg_loss', average_loss / 2000
            average_loss = 0
        if i % 10000 == 0:
            sim = similarity.eval()
            for j in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[j]]
                top_k = 8
                nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print log_str
    print 'start to save embeddings ....'
    final_embeddings = normalized_embeddings.eval()
    np.savetxt('final_embedding_dic.txt',final_embeddings)
    print  'finished saving'














    

    

