#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/18 12:38   gxrao      1.0         None
'''

from easydict import EasyDict as edict

v = edict()
v.absolute_path = r"F:\Jetbrains\python\HPO_LSTM"

v.random_state = 100

v.split = edict()
v.split.testSize = 0.95
v.split.num_times = 24
v.split.self = True

v.epochs = 200
v.batch_size = 512
v.num_workers = 20

v.input_dim = 24
v.hidden_dim = 100
v.layer_dim = 3
v.output_dim = 1
