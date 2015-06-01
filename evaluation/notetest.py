__author__ = 'rachel'

import evaluation.evaluate
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

folderpath = 'C:/Users/rachel/Desktop/emg/results_final/'
e = evaluation.evaluate.Evaluation()
e.results=pd.read_pickle('C:/Users/rachel/evaluation/notebooks/all_results')

e.make_same('file', ['angle', 'angle_small'])
e.make_same('file', ['stroke$_1$', 'stroke_left', 'stroke_P_4_self_left'])
e.make_same('file', ['stroke$_2$', 'stroke_right', 'stroke_P_4_self_right'])
e.make_same('file', ['pEMG', 'pemg', 'dataset-20140123-0R-a'])
e.make_same('file', ['action','physical'])
e.make_same('file', ['0G$_2$', '0g_tiny', '2011-05-10-vogel'])
e.make_same('file', ['0G$_1$', '0g', 'Hannes1'])
e.make_same('file', ['knee_old', 'bdsemg'])
e.make_same('file', ['knee', 'knee_1'])
e.make_same('file', ['knee$_3$', 'knee_3'])
e.make_same('file', ['knee$_5$', 'knee_5'])
e.make_same('model', ['pPCA', 'ppca'])
e.make_same('model', ['GMM', 'gmm'])
e.make_same('model', ['ICA', 'ica'])
e.make_same('model', ['FA', 'fa'])
e.make_same('model', ['MFA', 'mfa'])
e.make_same('model', ['VAE', 'VA', 'vae'])
e.make_same('model', ['MST', 'mst'])
e.make_same('function', ['lc','logcosh'])
e.make_same('function', ['lo','laplace'])
e.make_same('cov', ['diagonal', 'diag'])
e.make_same('fft', ['', -1, 0])
e.make_same('fft', ['fft', 1])
e.make_same('classification',['',-1,0])
e.make_same('classification',['class',1])

e.rename_attribute('window_size', 'window size')
e.rename_attribute('n_mixcomps', 'mixture components')
e.rename_attribute('n_features', 'features')
e.rename_attribute('n_latents','latents')

e.convert_flags_abbr()
e.set_order('window size', 'ascend')
e.set_order('mixture components', 'ascend')
e.set_order('latents','ascend')
e.set_order('model', ['pPCA','FA','ICA','VAE','GMM','MST','MFA'])
e.set_order('flags', ['raw', 's', 'w', 'sw', 'r', 'rs', 'rw', 'rsw'])
e.set_order('function', ['lc', 'lo'])

e.unfilter()
e.filter('model', 'pPCA')
e.filter('offset',10)
e.filter('file',['0G$_1$','0G$_2$','action','angle','knee','pEMG','stroke$_1$','stroke$_2$'])
e.filter('latents',[1,2,5,10,25,50])
#e.filter('window size', [1,2, 5, 10, 25, 50,100,200])
e.filter('fft','')
e.filter('classification','')
best = e.best_results_for(['file', 'flags'], fun='max')
counts = e.best_results_for(['file', 'flags'], fun='count')
#print counts
e.group_subplots(best, counts, error=True, no_rows=4, colormap='Pastel1', base=2, eps=.5)