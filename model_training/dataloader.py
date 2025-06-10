#Import Packages
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#Tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import utils

import glob

import random

class ECG_A1CDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.4.7")
    
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description='''
            Dataset for Estimating A1C from ECG Data
            ''',
            features=tfds.features.FeaturesDict({
                #Actual ECG
                "ecg": tfds.features.Tensor(
                  shape=(8,2500,), dtype=tf.float64),
                
                #PATIENT IDENTIFIERS
                "patient_id": tfds.features.Tensor(
                    shape=(), dtype=tf.int32),
                "encounter_id": tfds.features.Tensor(
                    shape=(), dtype=tf.int32),
                "date": tfds.features.Text(),
                
                #VISIT TYPE + TYPE
                "visittype": tfds.features.Text(),
                "type_": tfds.features.Text(),

                #A1C ORDER
                "collectioninstant": tfds.features.Text(),

                #ECG ORDER
                "seq_no": tfds.features.Text(),
                "ordertime": tfds.features.Text(),
                "timediff": tfds.features.Text(), 
                
                #LABELS
                "label": tfds.features.ClassLabel(num_classes=4),
                "value": tfds.features.Tensor(shape=(), dtype=tf.float64),
                
                #SAMPLE WEIGHTING
                "propensity": tfds.features.Tensor(shape=(), dtype=tf.float64),
                "n_encounters": tfds.features.Tensor(shape=(), dtype=tf.float64),
                "sample_weight": tfds.features.Tensor(shape=(), dtype=tf.float64),
                "sample_weight_clipped": tfds.features.Tensor(shape=(), dtype=tf.float64),

                "propensity_old": tfds.features.Tensor(shape=(), dtype=tf.float64),
                "sample_weight_old": tfds.features.Tensor(shape=(), dtype=tf.float64),
                "sample_weight_clipped_old": tfds.features.Tensor(shape=(), dtype=tf.float64),                

                #DEMOGRAPHICS (AGE/SMOKING/SEX/RACE)
                "age": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "sex": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "race": tfds.features.Tensor(
                  shape=(7,), dtype=tf.float64),
                "smoking": tfds.features.Tensor(
                  shape=(3,), dtype=tf.float64),
                
                "lifestyle": tfds.features.Tensor(
                  shape=(2,), dtype=tf.float64),
                "lifestyle_time": tfds.features.Tensor(
                  shape=(2,), dtype=tf.float64),
                "lifestyle_time_raw": tfds.features.Tensor(
                  shape=(2,), dtype=tf.float64),
                "age_raw":tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                

                "phys_act": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "phys_act_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                #FAMILY HX
                "fhx_cvd": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "fhx_dm": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                                
                #ECG_FEATURES
                "ecg_features": tfds.features.Tensor(
                  shape=(11,), dtype=tf.float64),
                "ecg_features_raw": tfds.features.Tensor(
                  shape=(11,), dtype=tf.float64),
                
                #VITALS
                "bmi": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "bp": tfds.features.Tensor(
                  shape=(2,), dtype=tf.float64),
                "bmi_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "bp_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "bmi_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "bp_raw": tfds.features.Tensor(
                  shape=(2,), dtype=tf.float64),
                "bmi_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "bp_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                
                #LABS
                "lipid": tfds.features.Tensor(
                  shape=(4,), dtype=tf.float64),
                "lipid_time": tfds.features.Tensor(
                  shape=(4,), dtype=tf.float64),
                "lipid_raw": tfds.features.Tensor(
                  shape=(4,), dtype=tf.float64),
                "lipid_time_raw": tfds.features.Tensor(
                  shape=(4,), dtype=tf.float64),
                
                "a1c": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "a1c_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "a1c_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "a1c_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                "egfr": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "egfr_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "egfr_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "egfr_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                #DIAGNOSES
                "t1dm": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "t1dm_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "t1dm_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                "t2dm": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "t2dm_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "t2dm_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                "gestational_dm": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "gestational_dm_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "gestational_dm_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                "pcos": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "pcos_time": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "pcos_time_raw": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                "dx": tfds.features.Tensor(
                  shape=(17,), dtype=tf.float64),
                "dx_time": tfds.features.Tensor(
                  shape=(17,), dtype=tf.float64),
                "dx_time_raw": tfds.features.Tensor(
                  shape=(17,), dtype=tf.float64),
                
                "acute_dx": tfds.features.Tensor(
                  shape=(7,), dtype=tf.float64),
                "acute_dx_time": tfds.features.Tensor(
                  shape=(7,), dtype=tf.float64),
                "acute_dx_time_raw": tfds.features.Tensor(
                  shape=(7,), dtype=tf.float64),
                
                #MEDICATIONS
                "rx": tfds.features.Tensor(
                  shape=(7,), dtype=tf.float64),
                
                #ADDITIONAL FEATURES 2211
                "ANTIHYPERTENSIVE": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "CORTICOSTEROID": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                "FAMHX_DM": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                "FAMHX_CVD": tfds.features.Tensor(
                  shape=(), dtype=tf.float64),
                
                
            }),
            supervised_keys=("ecg","label"),
        )

    def _split_generators(self, dl_manager):
        root_dir = '.'

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(split='train', root_dir=root_dir)),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs=dict(split='val', root_dir=root_dir)),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(split='test', root_dir=root_dir))
        ]

    def _generate_examples(self, split, root_dir):
        csv_path = os.path.join(root_dir, 'features_raw_230612_'+split+'.csv')
        with open(csv_path, 'r') as f:
            next(f)
            for i, line in enumerate(f.readlines()):
                record = line.rstrip().split(',')
                
                #PATIENT IDENTIFIERS
                patient_id = record[0]
                encounter_id = record[1]
                date = record[3]
                
                #VISIT TYPE + TYPE
                visittype = record[7]
                type_ = record[8]

                #A1C ORDER
                collectioninstant = record[4]

                #ECG ORDER
                seq_no = record[2]
                ordertime = record[5]
                timediff = record[6]
                
                #LABELS
                label = int(record[10])
                value = float(record[9])
                
                #SAMPLE WEIGHTING
                propensity_old = float(record[11])
                n_encounters = float(record[12])
                sample_weight_old = float(record[13])
                sample_weight_clipped_old = float(record[14])
                
                propensity = float(record[191])
                sample_weight = float(record[192])
                sample_weight_clipped = float(record[193])

                #DEMOGRAPHICS (AGE/SMOKING/SEX/RACE)
                age = float(record[15])
                sex = float(record[39])
                race = [float(n) for n in record[40:47]]
                smoking = [float(n) for n in record[36:39]]
                lifestyle = [float(record[51]), float(record[72])] 
                lifestyle_time = [float(record[51+47]), float(record[72+47])] 
                lifestyle_time_raw = [float(record[51+108]), float(record[72+108])] 
                age_raw = float(record[126])
                
                phys_act = float(record[72])
                phys_act_time = float(record[72+47])
                
                #FAMILY HX
                fhx_cvd = float(record[61])
                fhx_dm = float(record[62])
                                
                #ECG_FEATURES
                ecg_features = [float(n) for n in record[16:27]]
                ecg_features_raw = [float(n) for n in record[127:138]]
                
                #VITALS
                bmi =float(record[27])
                bp = [float(n) for n in record[28:30]]
                bmi_time = float(record[27+59])
                bp_time = float(record[28+59])
                bmi_raw = float(record[27+111])
                bp_raw = [float(n) for n in record[28+111:30+111]]
                bmi_time_raw = float(record[27+120])
                bp_time_raw = float(record[28+120])
                
                #LABS
                lipid = [float(n) for n in [record[31]] + record[33:36]]
                lipid_time = [float(n) for n in [record[31+58]] + record[33+58:36+58]]
                lipid_raw = [float(n) for n in [record[31+111]] + record[33+111:36+111]]
                lipid_time_raw = [float(n) for n in [record[31+119]] + record[33+119:36+119]]
                
                a1c = float(record[30])
                a1c_time = float(record[30+58])
                a1c_raw = float(record[30+111])
                a1c_time_raw = float(record[30+119])
                
                egfr = float(record[32])
                egfr_time = float(record[32+58])
                egfr_raw = float(record[32+111])
                egfr_time_raw = float(record[32+119])
                
                #DIAGNOSES
                t1dm = float(record[58])
                t1dm_time = float(record[58+47])
                t1dm_time_raw = float(record[58+108])
                
                t2dm = float(record[59])
                t2dm_time = float(record[59+47])
                t2dm_time_raw = float(record[59+108])
                
                gestational_dm = float(record[63])
                gestational_dm_time = float(record[63+47])
                gestational_dm_time_raw = float(record[63+108])
                
                pcos = float(record[71])
                pcos_time = float(record[71+47])
                pcos_time_raw = float(record[71+108])
                
                dx = [float(n) for n in record[47:50] + record[53:58] + [record[60]] + record[64:71] + [record[76]]]
                dx_time = [float(n) for n in record[47+47:50+47] + record[53+47:58+47] + [record[60+47]] + 
                           record[64+47:71+47] + [record[76+47]]]
                dx_time_raw = [float(n) for n in record[47+108:50+108] + record[53+108:58+108] + [record[60+108]] + 
                               record[64+108:71+108] + [record[76+108]]]
                
                acute_dx = [float(n) for n in [record[50]] + [record[52]] + record[73:76] + record[77:79]]
                acute_dx_time = [float(n) for n in [record[50+47]] + [record[52+47]] + 
                                 record[73+47:76+47] + record[77+47:79+47]]
                acute_dx_time_raw = [float(n) for n in [record[50+108]] + [record[52+108]] + 
                                     record[73+108:76+108] + record[77+108:79+108]]
                
                #MEDICATIONS
                rx = [float(n) for n in record[79:86]]
                
                #ADDITIONAL FEATURES 2211
                ANTIHYPERTENSIVE = float(record[187])
                CORTICOSTEROID = float(record[188])
                FAMHX_DM = float(record[189])
                FAMHX_CVD = float(record[190])
                
                #Read in ECG
                ecg_path = os.path.join('./', 'ecgs', str(seq_no) +'.npy')
                ecg = np.load(ecg_path)
                
                yield encounter_id, dict(patient_id = patient_id, encounter_id = encounter_id, 
                     date = date, visittype = visittype, type_ = type_, 
                     collectioninstant = collectioninstant, seq_no = seq_no, 
                     ordertime = ordertime, timediff = timediff, label = label, 
                     value = value, propensity = propensity, n_encounters = n_encounters, 
                     sample_weight = sample_weight, sample_weight_clipped = sample_weight_clipped, 
                    age = age, sex = sex, race = race, smoking = smoking, 
                    lifestyle = lifestyle, lifestyle_time = lifestyle_time, lifestyle_time_raw = lifestyle_time_raw, 
                    age_raw = age_raw, fhx_cvd = fhx_cvd, fhx_dm = fhx_dm, 
                    bmi =bmi, bp = bp, bmi_time = bmi_time, bp_time = bp_time, 
                    bmi_raw = bmi_raw, bp_raw = bp_raw, bmi_time_raw = bmi_time_raw, bp_time_raw = bp_time_raw, 
                    lipid = lipid, lipid_time = lipid_time, lipid_raw = lipid_raw, lipid_time_raw = lipid_time_raw, 
                    a1c = a1c, a1c_time = a1c_time, a1c_raw = a1c_raw, a1c_time_raw = a1c_time_raw, 
                    egfr = egfr, egfr_time = egfr_time, egfr_raw = egfr_raw, egfr_time_raw = egfr_time_raw, 
                    t1dm = t1dm, t1dm_time = t1dm_time, t1dm_time_raw = t1dm_time_raw, 
                    t2dm = t2dm, t2dm_time = t2dm_time, t2dm_time_raw = t2dm_time_raw, 
                    gestational_dm = gestational_dm, gestational_dm_time = gestational_dm_time, 
                    gestational_dm_time_raw = gestational_dm_time_raw, 
                    pcos = pcos, pcos_time = pcos_time, pcos_time_raw = pcos_time_raw, 
                    dx = dx, dx_time = dx_time, dx_time_raw = dx_time_raw,            
                    acute_dx = acute_dx, acute_dx_time = acute_dx_time,               
                    acute_dx_time_raw = acute_dx_time_raw,
                     ecg_features = ecg_features,ecg_features_raw = ecg_features_raw,
                                         rx = rx, 
                                         ANTIHYPERTENSIVE=ANTIHYPERTENSIVE, CORTICOSTEROID=CORTICOSTEROID,
                                         FAMHX_DM=FAMHX_DM, FAMHX_CVD=FAMHX_CVD, 
                                         propensity_old=propensity_old, sample_weight_old=sample_weight_old, 
                                         sample_weight_clipped_old=sample_weight_clipped_old, 
                                         phys_act=phys_act, phys_act_time=phys_act_time,
                                         ecg=ecg)