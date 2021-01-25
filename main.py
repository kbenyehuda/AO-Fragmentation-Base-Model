import numpy as np
import pickle
import pandas as pd
import cv2
from reading_feats import *
from set_weights_for_model import *
from preprocesing_feats import *
from preprocessing_pics import *
from model_7_feats import *

base_path = 'D:/Users/Keren/Documents/university/Year 2/DNA Fragmentation/Acridine Orange/feature_extractions/'

all_dfs,donors = read_feats_file(base_path+'final_exp_3.xlsx')

combined_model = set_weights('New_AO_combined_7feats_2best_prec.pickle',combined_model)

image_location_base_dir = 'E:/Keren Sperm Project Backup/final_exp'

all_ao_outputs = {}

for i in range(len(all_dfs)):
  all_images = []
  all_feats = []
  cur_indexes = list(all_dfs[i].iloc[:,0])
  for j in range(len(cur_indexes)):
    cur_index = cur_indexes[j]
    v_ind = cur_index.index('v')
    f_ind = cur_index.index('f')
    C_ind = cur_index.index('C')

    don_num = (cur_index[1:v_ind])
    vid_num = (cur_index[v_ind + 1:f_ind])
    frm_num = (cur_index[f_ind + 1:C_ind])
    cel_num = (cur_index[C_ind + 2:])

    cur_location = image_location_base_dir + '/donor ' + don_num + '/video_' + vid_num + '/C_' + cel_num + '/' + frm_num + '.png'
    all_images.append(resize(cv2.imread(cur_location)))
    all_feats.append(np.array(all_dfs[i].iloc[j, 1:]))

  all_images = np.array(all_images)
  all_images = all_images[:,:,:,0]
  all_images = all_images[:,:,:,np.newaxis]
  all_feats = np.array(all_feats)
  all_feats = normalizing_features(all_feats)
  cur_AO_output = combined_model.predict([all_images, all_feats],batch_size=32,verbose=1)

  cur_df = all_dfs[i]
  cur_df['AO'] = cur_AO_output
  all_ao_outputs[donors[i]]=np.nanmean(cur_AO_output)


# with pd.ExcelWriter(base_path+'final_exp_wao.xlsx') as writer:
#     columns = ['Total Head Area','Nucleus Area','Acrosome Area','Mean Post Ant Diff','Mean OPD','Var OPD','Drymass','AO']
#     for i in range(len(all_dfs)):
#         df = all_dfs[i]
#         sheet_name = 'donor '+str(donors[i])
#         df.to_excel(writer, sheet_name=sheet_name)
