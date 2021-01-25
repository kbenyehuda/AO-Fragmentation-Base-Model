from reading_feats import *
import pandas as pd
import numpy as np

base_path = 'D:/Users/Keren/Documents/university/Year 2/DNA Fragmentation/Acridine Orange/feature_extractions/'
all_dfs,donors = read_feats_file(base_path+'final_exp_wao.xlsx')
# for every sheet get cell name and take num_frames, min max mean median and std

all_donors_cells = {}

for i in range(len(all_dfs)):
    print('i: ',i)
    all_cells_dict = {}
    cur_indexes = list(all_dfs[i].iloc[:, 1])
    for j in range(len(cur_indexes)):
        cur_index = cur_indexes[j]
        C_ind = cur_index.index('C')
        cel_num = (cur_index[C_ind + 2:])
        AO_pred = all_dfs[i].iloc[j,-1]
        if cel_num not in all_cells_dict:
            all_cells_dict[cel_num]=[AO_pred]
        else:
            all_cells_dict[cel_num].append(AO_pred)
    all_cells = list(all_cells_dict.keys())
    new_cells_dict = {}
    for k in range(len(all_cells)):
        try:
            cur_cell = all_cells[k]
            cur_cell_preds = all_cells_dict[cur_cell]
            # cur_cell_ao_stats = [len(cur_cell_preds),np.nanmin(cur_cell_preds),np.nanmean(cur_cell_preds),
            #                      np.nanmedian(cur_cell_preds),np.nanmax(cur_cell_preds),np.std(cur_cell_preds)]
            cur_cell_ao_stats = np.nanmin(cur_cell_preds)
            new_cells_dict[cur_cell]=cur_cell_ao_stats
        except:
            continue
    all_donors_cells[donors[i]]=new_cells_dict



# import pickle
# with open('all_donors_cells.pickle', 'wb') as f:
#   pickle.dump(all_donors_cells, f)


# with pd.ExcelWriter(base_path+'ao_stats.xlsx') as writer:
#     for i in range(len(donors)):
#         cur_dict = all_donors_cells[donors[i]]
#         cur_df = pd.DataFrame(cur_dict).T
#         sheet_name = 'donor ' + str(donors[i])
#         cur_df.to_excel(writer, sheet_name=sheet_name)



