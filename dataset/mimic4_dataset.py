import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.tokenizer import MIMIC4Tokenizer
from fed_experiments.utils import processed_data_path, read_txt, load_pickle, split_list_multi_parts, mimic_missing_status


class MIMIC4Dataset(Dataset):
    def __init__(self, split, task, client_idx, num_map, load_no_label=False, dev=False, return_raw=False, miss_rate=0.0):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        # load complete data
        all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "mimic4/hosp_adm_dict_v2.pkl"))
        all_included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"mimic4/task:{task}/{split}_admission_ids.txt"))
        total_num = len(all_included_admission_ids)
        # local partition
        local_idx = split_list_multi_parts(list(range(total_num)), num_map)[client_idx]
        local_include_admission_ids = [all_included_admission_ids[i] for i in local_idx]
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"mimic4/task:{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            local_include_admission_ids += no_label_admission_ids
        self.local_include_admission_ids = local_include_admission_ids
        if dev:
            self.local_include_admission_ids = self.local_include_admission_ids[:10000]
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()
        # store
        self.local_data = []
        for admission_id in self.local_include_admission_ids:
            hosp_adm = all_hosp_adm_dict[admission_id]
            self.local_data.append(hosp_adm)
        # data status
        miss_mask = mimic_missing_status(self.local_data, split, task, client_idx, True)
        if miss_rate:
            ratio_discharge = 0.72
            ratio_lab = 1 - ratio_discharge
            mod_mask = miss_mask[0]
            target_miss_nums = int(np.size(mod_mask) * miss_rate)
            current_miss_nums = np.sum(mod_mask)
            delta = max(0, target_miss_nums - current_miss_nums)
            if delta:
                prob_pos = np.argwhere(mod_mask == False)
                dis_candidate_idx = []
                lab_candidate_idx = []
                for item in prob_pos:
                    if item[1] == 0:
                        dis_candidate_idx.append(item)
                    elif item[1] == 1:
                        lab_candidate_idx.append(item)
                if len(dis_candidate_idx) > int(ratio_discharge * delta):
                    selected_dis_idx = np.random.choice(len(dis_candidate_idx), int(ratio_discharge * delta), replace=False)
                else:
                    selected_dis_idx = list(range(len(dis_candidate_idx)))
                if len(lab_candidate_idx) > int(ratio_lab * delta):
                    selected_lab_idx = np.random.choice(len(lab_candidate_idx), int(ratio_lab * delta), replace=False)
                else:
                    selected_lab_idx = list(range(len(lab_candidate_idx)))
                for idx in selected_dis_idx:
                    entity_idx = dis_candidate_idx[idx][0]
                    self.local_data[entity_idx].discharge = None
                for idx in selected_lab_idx:
                    entity_idx = lab_candidate_idx[idx][0]
                    self.local_data[entity_idx].labvectors = None
            # double check
            print("===data dropping on client {}===".format(client_idx))
            mimic_missing_status(self.local_data, split, task, client_idx, True)

    def __len__(self):
        return len(self.local_include_admission_ids)

    def __getitem__(self, index):
        admission_id = self.local_include_admission_ids[index]
        hosp_adm = self.local_data[index]

        age = str(hosp_adm.age)
        gender = hosp_adm.gender
        ethnicity = hosp_adm.ethnicity
        types = hosp_adm.trajectory[0]
        codes = hosp_adm.trajectory[1]
        codes_flag = True

        labvectors = hosp_adm.labvectors
        labvectors_flag = True
        if labvectors is None:
            labvectors = torch.zeros(1, 116)
            labvectors_flag = False
        else:
            labvectors = torch.FloatTensor(labvectors)

        discharge = hosp_adm.discharge
        discharge_flag = True
        if discharge is None:
            discharge = ""
            discharge_flag = False

        label = getattr(hosp_adm, self.task)
        label_flag = True
        if label is None:
            label = 0.0
            label_flag = False
        else:
            label = float(label)

        if not self.return_raw:
            age, gender, ethnicity, types, codes = self.tokenizer(
                age, gender, ethnicity, types, codes
            )
            label = torch.tensor(label)

        return_dict = dict()
        return_dict["id"] = admission_id

        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["types"] = types
        return_dict["codes"] = codes
        return_dict["codes_flag"] = codes_flag

        return_dict["labvectors"] = labvectors
        return_dict["labvectors_flag"] = labvectors_flag

        return_dict["discharge"] = discharge
        return_dict["discharge_flag"] = discharge_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict
