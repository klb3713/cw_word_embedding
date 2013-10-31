# -*- coding: utf-8 -*-
__author__ = 'klb3713'

import os
import re
import zipfile
import config


def get_zip_files(dirs):
    zip_files = []
    for dir in dirs:
        for parent,dirnames,filenames in os.walk(dir):
            for filename in filenames:
                if filename.endswith('.zip'):
                    zip_files.append(os.path.join(parent, filename))

    return zip_files

p_word = re.compile(r'[a-zA-Z]+')
p_para = re.compile(r'(?<=<p>)[^<]+(?=</p>)')

def read_zip_files(zip_files):
    corpus = open(os.path.join(config.DATA_DIR, 'corpus'), 'w')
    for file in zip_files:
        try:
            z = zipfile.ZipFile(file, "r")

            for filename in z.namelist():
                print('File: %s: %s' %(file, filename))

                #读取zip文件中的文件
                content = z.read(filename)
                paras = p_para.findall(content)
                for para in paras:
                    if len(p_word.findall(para)) < config.WINDOW_SIZE:
                        continue
                    para = para.replace('&quot;', '')
                    corpus.write(para + '\n')
        except Exception, e:
            print(file, e)

    corpus.close()


p_word = re.compile(r'[a-zA-Z0-9]+')

def getNormalWord(word):
    if word.isdigit():
        return 'SYMBOL'
    elif word.isalnum():
        return word.lower()

    word = ''.join(p_word.findall(word))
    if word.isdigit():
        return 'SYMBOL'
    else:
        return word.lower()

p_sent = re.compile(r'[?!;.]\s')

def get_pure_corpus(corpus_file):
    corpus = open(corpus_file, 'w')
    with open(config.TRAIN_FILE, 'r') as f:
        for line in f:
            line = line.strip('\n')
            sents = p_sent.split(line)

            for sent in sents:
                words = [getNormalWord(word) for word in sent.split()]
                corpus.write(' '.join(words) + '\n')

    corpus.close()


if __name__ == "__main__":
    # dirs = ['/home/klb3713/workspace/resource/reuters_corpus/110104_0810',
    #         '/home/klb3713/workspace/resource/reuters_corpus/100623_0655']
    # zip_files = get_zip_files(dirs)
    # read_zip_files(zip_files)

    get_pure_corpus('data/pure_RC_corpus')