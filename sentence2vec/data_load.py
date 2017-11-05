#!encoding=utf8
import numpy as np
import collections, itertools
import os, math, jieba, random, logging, codecs


class DataLoader(object):
    
    def __init__(self, data_file, data_file_format='ascii', vocabulary_size=50000, stop_word_file=None, use_jieba=False):
        self.data_file = data_file
        self.data_file_format = data_file_format
        self.stop_word_file = stop_word_file
        self.vocabulary_size = vocabulary_size
        self.data_index = 0
        self.doc_ids = None
        self.word_ids = None
        self.dictionary = None
        self.reserved_dictionary = None
        self.sentence_list = None
        self.line_list = None
        self.use_jieba = use_jieba
    
    def load(self):
        self.sentence_list, self.line_list = self.read_data(self.data_file, self.stop_word_file)
        print('sentence len ', len(self.sentence_list))

        self.doc_ids, self.word_ids, _, self.dictionary, self.reversed_dictionary = self.build_docs(self.sentence_list, self.vocabulary_size)
        print 'word cnt:', len(self.word_ids), 'word_max', max(self.word_ids)
  
    def read_data(self, data_file, stop_word_file):
        """
        对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
        """
        #读取停用词
        stop_words = []
        if stop_word_file:
            with codecs.open(stop_word_file, "r", self.data_file_format) as f:
                line = f.readline()
                while line:
                    stop_words.append(line[:-1])
                    line = f.readline()
            stop_words = set(stop_words)
            print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    
        # 读取文本，预处理，分词，得到词典
        sentence_list = []
        line_list = []
        print 'using jieba:', self.use_jieba
        with codecs.open(data_file, "r", self.data_file_format) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if len(line)>0: # 如果句子非空
                    if self.use_jieba:
                        while ' ' in line:
                            line = line.replace(' ','')
                        raw_words = list(jieba.cut(line, cut_all=False))
                    else:
                        raw_words = line.split()
                    words = [w.strip() for w in raw_words if w not in stop_words and len(w.strip())>0 ]
                    if len(words) < 3:
                        continue
                    sentence_list.append(words)
                    line_list.append(line)
        return sentence_list, line_list
    
        
    def build_dataset(self, words, vocabulary_size):
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
    
    
    def build_docs(self, sentence_list, vocabulary_size):
        sentence_ids = []
        words = []
        for i, sentence in enumerate(sentence_list):
            sentence_ids.extend([i] * len(sentence))
            words.extend(sentence)
        word_ids, count, dictionary, reversed_dictionary = self.build_dataset(words, vocabulary_size)
        return sentence_ids, word_ids, count, dictionary, reversed_dictionary
    
    def next_step(self, max_step):
        assert max_step >= 1
        return random.randint(1, max_step)

    def generate_batch_pvdm(self, batch_size, window_size):
        assert batch_size % window_size == 0
        doc_ids = self.doc_ids
        word_ids = self.word_ids
        batch = np.ndarray(shape=(batch_size, window_size+1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = window_size + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        buffer_doc = collections.deque(maxlen=span)
        
        buffer.extend(word_ids[self.data_index: self.data_index+span])
        buffer_doc.extend(doc_ids[self.data_index: self.data_index+span])
        n_step = self.next_step(max_step=min(window_size, 1))
        self.data_index = (self.data_index + n_step) % len(word_ids)
        mask = [1] * span
        mask[-1] = 0
        i = 0
        while i < batch_size:
            if len(set(buffer_doc)) == 1:
                ## one sentence has enough words (window_size+1 words)
                ## batch is [worc_vec1, ... word_vecN, doc_vec]
                doc_id = buffer_doc[-1]
                batch[i, :] = list(itertools.compress(buffer, mask)) + [doc_id]
                labels[i, 0] = buffer[-1]
                i += 1
            buffer.extend(word_ids[self.data_index: self.data_index+span])
            buffer_doc.extend(doc_ids[self.data_index: self.data_index+span])
            n_step = self.next_step(max_step=min(window_size, 1))
            self.data_index = (self.data_index + n_step) % len(word_ids)
        return batch, labels

    def generate_skip_through_batch(self, batch_size):
        
        def get_batch_length(batch):
            return [len(b) for b in batch]
        def get_vector(row):
            return [self.dictionary[w] for w in row if w in self.dictionary]

        def to_full_batch(batch):
            max_len = max(get_batch_length(batch))
            batch_size = len(batch)
            full_batch = np.full((batch_size, max_len), 0, np.int32)
            for row in xrange(batch_size):
                full_batch[row, :len(batch[row])] = batch[row]
            return full_batch

        assert len(self.sentence_list) > 3
        if self.data_index == 0:
            self.data_index += 1

        batch_cur = []
        batch_pre = []
        batch_post = []
        i = 0
        while i < batch_size:
            if self.data_index + 1 >= len(self.sentence_list):
                self.data_index = 1
            batch_cur.append(get_vector(self.sentence_list[self.data_index]))
            batch_pre.append(get_vector(self.sentence_list[self.data_index-1]))
            batch_post.append(get_vector(self.sentence_list[self.data_index+1]))
            self.data_index += 1
            i += 1
        
        batch_inputs = batch_cur
        batch_inputs_len = get_batch_length(batch_inputs)
        

        batch_pre_inputs = [x[:-1] for x in batch_pre]
        batch_pre_inputs_len = get_batch_length(batch_pre_inputs)
        batch_pre_targets = [x[1:] for x in batch_pre]

        batch_post_inputs = [x[:-1] for x in batch_post]
        batch_post_inputs_len = get_batch_length(batch_post_inputs)
        batch_post_targets = [x[1:] for x in batch_post]

        return to_full_batch(batch_inputs), batch_inputs_len, to_full_batch(batch_pre_inputs), batch_pre_inputs_len, to_full_batch(batch_pre_targets), to_full_batch(batch_post_inputs), batch_post_inputs_len, to_full_batch(batch_post_targets)

