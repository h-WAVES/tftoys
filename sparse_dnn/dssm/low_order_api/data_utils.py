#coding=utf-8
import sys
import os
import random
import datetime
import time
# import _pickle as pickle
import json
import numpy as np
from cityhash import CityHash64

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

vocab = {}

cur_running_path = '/logging/qichao/lookalike/dssm'

# insntace format: Gender AgeClass City Platform XssCategory_1-7 UcwebCategory_1-7 QueryCategory_1-3 TaobaoCategory_1-3 XssTag_1-3 UcwebTag_1-3 App_1-10(过滤微信电商团购地图等)
#                  Tmall_ Taobao_ Car_ House_ Click_1-5 Conv_1-3 TopAccountCtr_1-3 TopAccountCvr_1-3 TopTopicCtr_1-3 TopTopicCvr_1-3 

def load_creative_feature():
  train_path = '%s/data/creative2feature' % cur_running_path
  if os.path.exists(train_path):
    return json.load(open(train_path))
  else:
    print('error when load creative feature')
    exit(-1)
creative2feature = load_creative_feature()

def load_all_dataset(day_idx):
  train_path = '%s/data/train.txt.%s'%(cur_running_path, day_idx)
  if os.path.exists(train_path):
    d = [i for i in open(train_path).readlines() if '\x01' in i]
    d = sorted(d, key=lambda x:x.strip().split('\x01')[1], reverse=True)
    return d
  return None

def load_dataset_batch(data):
  global creative2feature
  vocab_size = FLAGS.vocab_size
  query = []
  doc = []
  label = []
  for row in data:
  #for row in open(filepath).readlines():
    try:
      row = row.strip('\n').split('\x01')
      if len(row) != 6:
        print('wrong format,len:%d %s'%(len(row), row))
        continue
      '''sid uid ideaid stimestamp dssmfeature roilabel'''
      query_list = row[4].split(',')
      # if (len(query_list) != 165) or (row[2] not in creative2feature):
      if len(query_list) != 165:
        continue
      query_list = [int(word) for word in query_list]
      new_query_list = query_list
    except(Exception) as e:
      continue
    query.append(new_query_list)
    # doc.append(creative2feature[row[2]])
    doc.append(creative2feature["51895429"])
    label.append([1 if row[-1] != '0' else 0])
  if len(query) != 0 and len(query) == len(doc) and len(query) == len(label):
    return np.asarray(query, dtype=np.int32), np.asarray(doc, dtype=np.int32), np.asarray(label, dtype=np.int32)
  else:
    return None, None, None
