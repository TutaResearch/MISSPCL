import os

import numpy as np
import torch
from torch.utils.data import Dataset

from fed_experiments.utils import adni_missing_status, read_txt, load_pickle, split_list_multi_parts

# your processed dataset path
processed_data_path = r'/tadpole_challenge/processed/'


class ADNIDataset(Dataset):
    def __init__(self, split, task, client_idx, num_map, load_no_label=False, dev=False, miss_rate=0.0):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        # load complete data
        all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "adni_data_dict.pkl"))
        all_included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"task:{task}/{split}_admission_ids.txt"))
        total_num = len(all_included_admission_ids)
        # local partition
        local_idx = split_list_multi_parts(list(range(total_num)), num_map)[client_idx]
        local_include_admission_ids = [all_included_admission_ids[i] for i in local_idx]
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"task:{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            local_include_admission_ids += no_label_admission_ids
        self.local_include_admission_ids = local_include_admission_ids
        if dev:
            self.local_include_admission_ids = self.local_include_admission_ids[:10000]
        # store
        self.local_data = []
        for admission_id in self.local_include_admission_ids:
            hosp_adm = all_hosp_adm_dict[admission_id]
            self.local_data.append(hosp_adm)
        # data status
        miss_mask = adni_missing_status(self.local_data, split, task, client_idx, True)
        if miss_rate:
            ratio_x1 = 0.5
            ratio_x2 = 0.35
            ratio_x3 = 1 - ratio_x1 - ratio_x2
            mod_mask = miss_mask[0]
            target_miss_nums = int(np.size(mod_mask) * miss_rate)
            current_miss_nums = np.sum(mod_mask)
            delta = max(0, target_miss_nums - current_miss_nums)
            if delta:
                prob_pos = np.argwhere(mod_mask == False)
                x1_candidate_idx = []
                x2_candidate_idx = []
                x3_candidate_idx = []
                for item in prob_pos:
                    if item[1] == 0:
                        x1_candidate_idx.append(item)
                    elif item[1] == 1:
                        x2_candidate_idx.append(item)
                    elif item[1] == 2:
                        x3_candidate_idx.append(item)
                if len(x1_candidate_idx) > int(ratio_x1 * delta):
                    x1_selected_idx = np.random.choice(len(x1_candidate_idx), int(ratio_x1 * delta), replace=False)
                else:
                    x1_selected_idx = list(range(len(x1_candidate_idx)))
                if len(x2_candidate_idx) > int(ratio_x2 * delta):
                    x2_selected_idx = np.random.choice(len(x2_candidate_idx), int(ratio_x2 * delta), replace=False)
                else:
                    x2_selected_idx = list(range(len(x2_candidate_idx)))
                if len(x3_candidate_idx) > int(ratio_x3 * delta):
                    x3_selected_idx = np.random.choice(len(x3_candidate_idx), int(ratio_x3 * delta), replace=False)
                else:
                    x3_selected_idx = list(range(len(x3_candidate_idx)))
                for idx in x1_selected_idx:
                    entity_idx = x1_candidate_idx[idx][0]
                    self.local_data[entity_idx].x1 = None
                for idx in x2_selected_idx:
                    entity_idx = x2_candidate_idx[idx][0]
                    self.local_data[entity_idx].x2 = None
                for idx in x3_selected_idx:
                    entity_idx = x3_candidate_idx[idx][0]
                    self.local_data[entity_idx].x3 = None
            # double check
            print("===data dropping on client {}===".format(client_idx))
            adni_missing_status(self.local_data, split, task, client_idx)

    def __len__(self):
        return len(self.local_include_admission_ids)

    def __getitem__(self, index):
        icu_id = self.local_include_admission_ids[index]
        icu_stay = self.local_data[index]

        x1 = icu_stay.x1
        x1_flag = True
        if x1 is None:
            x1 = torch.zeros(228)
            x1_flag = False
        else:
            x1 = torch.FloatTensor(x1)

        x2 = icu_stay.x2
        x2_flag = True
        if x2 is None:
            x2 = torch.zeros(227)
            x2_flag = False
        else:
            x2 = torch.FloatTensor(x2)

        x3 = icu_stay.x3
        x3_flag = True
        if x3 is None:
            x3 = torch.zeros(620)
            x3_flag = False
        else:
            x3 = torch.FloatTensor(x3)

        label = getattr(icu_stay, self.task)
        label_flag = True
        if label is None:
            label = 0
            label_flag = False
        else:
            label = int(label)

        return_dict = dict()
        return_dict["id"] = icu_id

        return_dict["x1"] = x1
        return_dict["x1_flag"] = x1_flag

        return_dict["x2"] = x2
        return_dict["x2_flag"] = x2_flag

        return_dict["x3"] = x3
        return_dict["x3_flag"] = x3_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict
