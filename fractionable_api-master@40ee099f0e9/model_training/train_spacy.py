# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:28:27 2023

@author: ravit
"""
import s3fs
import time
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
from IPython import get_ipython
# from IPython import get_ipython
from pathlib import Path
from spacy.cli.download import download
from spacy.cli.init_config import fill_config
from spacy.cli.train import train as spacy_train
import os, glob
import spacy
import mlflow
import mlflow.spacy
import json
import configparser
import pandas as pd
# import boto3
import json
import boto3
import sys
import logging
from operator import itemgetter
from datetime import datetime, timedelta, timezone
from dateutil.tz import tzutc
# import mlflow
import configparser
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from spacy.training import offsets_to_biluo_tags


def get_spacy_doc(file, data):
    # import pdb; pdb.set_trace()
    nlp = spacy.blank('en')
    db = DocBin()
    for text, annot in tqdm(data):
        doc = nlp.make_doc(text)
        annot = annot['entities']

        ents = []
        entity_indices = []

        for start, end, label in annot:
            skip_entity = False
            for idx in range(start, end):
                if idx in entity_indices:
                    skip_entity = True
                    break
            if skip_entity == True:
                continue

            entity_indices = entity_indices + list(range(start, end))

            try:
                span = doc.char_span(start, end, label=label, alignment_mode='strict')
            except:
                continue

            if span is None:
                err_data = str([start, end]) + " " + str(text) + "\n"
                # file.write(err_data)

            else:
                ents.append(span)

        try:
            doc.ents = ents
            db.add(doc)
        except:
            pass
    return db


def get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label


def create_total_target_vector(docs):
    target_vector = []
    for doc in docs:
        #print(doc)
        new = nlp.make_doc(doc[0])
        entities = doc[1]["entities"]
        bilou_entities = offsets_to_biluo_tags(new, entities)
        final = []
        for item in bilou_entities:
            final.append(get_cleaned_label(item))
        target_vector.extend(final)
    return target_vector


def create_prediction_vector(text):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text)]


def create_total_prediction_vector(docs: list):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc[0]))
    return prediction_vector


def get_all_ner_predictions(text):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities


def get_model_labels():
    labels = list(nlp.get_pipe("ner").labels)
    labels.append("O")
    return sorted(labels)


def get_dataset_labels(docs):
    return sorted(set(create_total_target_vector(docs)))


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def generate_confusion_matrix(docs):
    classes = sorted(set(create_total_target_vector(docs)))
    y_true = create_total_target_vector(docs)
    y_pred = create_total_prediction_vector(docs)
    # print(classification_report(y_true, y_pred))
    # print("cohen_kappa_score:",cohen_kappa_score(y_true,y_pred))
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    # print (y_true)
    # print (y_pred)
    return confusion_matrix(y_true, y_pred), cohen_kappa

config = configparser.ConfigParser()
config.optionxform = str
path = "/home/ptgml/resume/CV-Parsing-using-Spacy-3/data/training/config.cfg"
config.read(path)
client = boto3.client(service_name='s3', region_name='ap-south-1', aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
                      aws_secret_access_key='c+zQLMpoaLUMNqjF1t/pSHaXgwROSPcQsmXcUKjP')
result = client.get_object(Bucket='fractionable', Key='actualdata/samplejsons/confi.json')
content = result['Body'].read().decode()
configurable_json = json.loads(content)
print(configurable_json)
config.sections()
# print(config["training.optimizer"]['2_is_weight_decay'])
"""config['training']['max_steps']=str(content['max_steps'])
config['training']['eval_frequency']=str(content['eval_frequency'])
config['nlp']['batch_size']=str(content['batch_size'])
config['training.optimizer']['@optimizers']=str(content['optimizer'])
config['training.optimizer.learn_rate']['initial_rate']=str(content['inital_rate'])
config['components.transformer.model']['name']=str(content['model_name'])
with open(path, 'w') as f:
     config.write(f)"""

number = 5
VERSION = 1.0
# print(content['resume_sections'])
resume_sections_list = configurable_json['resume_sections']
#print(resume_sections_list)
for section in resume_sections_list:
    #import pdb; pdb.set_trace()
    # if list1=="combined":
    # s3 = boto3.client('s3')
    """s3 = boto3.resource(
        service_name='s3',
        region_name='ap-south-1',
        aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
        aws_secret_access_key='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')
    instances=['i-0d583327b7b3ed767']
    ec2_resource = boto3.resource('ec2', region_name='ap-south-1')
    ec2 = boto3.client('ec2', region_name='ap-south-1')
    instance = ec2_resource.Instance('i-0d583327b7b3ed767')"""
    # s3 = boto3.client('s3')
    client = boto3.client('s3', region_name='ap-south-1',
                          aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
                          aws_secret_access_key='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')
    response = client.list_objects_v2(
        Bucket="fractionable",
        Prefix=f"val_argilla1/{section}")
    response_2 = client.list_objects_v2(
        Bucket="fractionable",
        Prefix=f"datasets/{section}")
    #import pdb; pdb.set_trace()
    if len(response_2['Contents'][1:]) > 0:
        last_trained_date = max(response_2['Contents'], key=lambda x: x['LastModified'])['LastModified']
        highest_kappa = float(response_2['Contents'][1:][0]['Key'].split("/")[-1].split("_")[-1].replace(".json", ""))
        cv_res = client.get_object(Bucket='fractionable', Key=sorted(response_2['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]['Key'])
        cv_content = cv_res['Body'].read()
        cv_datasets = json.loads(cv_content)
    else:
        last_trained_date = datetime(2022, 4, 25, 11, 23, 21, tzinfo=tzutc())
        highest_kappa = 0
        cv_datasets = []
    print("previous_cv_data_count", len(cv_datasets))
    #last_trained_date=datetime(2022, 4, 25, 11, 23, 21, tzinfo=tzutc())
    resumes_count_after_last_trained_date = []
    for key in (response['Contents'][1:]):
        #import pdb; pdb.set_trace()
        if key['LastModified'] >= last_trained_date:
            resumes_count_after_last_trained_date.append(key['Key'])
            # len(list1)

    #import pdb; pdb.set_trace()
    print("---------------------------------------------------------------------------------")
    print(section,len(resumes_count_after_last_trained_date))
    if len(resumes_count_after_last_trained_date) >= number:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # number = 0
        VERSION = 1.0
        # s3 = boto3.client('s3')
        """s3 = boto3.resource(
            service_name='s3',
            region_name='ap-south-1',
            aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
            aws_secret_access_key='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')
        instances=['i-0d583327b7b3ed767']
        ec2_resource = boto3.resource('ec2', region_name='ap-south-1')
        ec2 = boto3.client('ec2', region_name='ap-south-1')
        instance = ec2_resource.Instance('i-0d583327b7b3ed767')"""
        # s3 = boto3.client('s3')
        client = boto3.client(service_name='s3',
                              region_name='ap-south-1',
                              aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
                              aws_secret_access_key='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')
        response = client.list_objects_v2(
            Bucket="fractionable",
            Prefix=f"val_argilla1/{section}")
        list_of_resumes = []
        for res in response['Contents'][1:]:
            if res['LastModified'] >= last_trained_date:
                res_1 = client.get_object(Bucket='fractionable', Key=res['Key'])
                content = res_1['Body']
                jsonObject = json.loads(content.read())
                cv_datasets.append(jsonObject)
        # import pdb; pdb.set_trace()
        print("updated_cv_datasets_count", len(cv_datasets))
        train,test = train_test_split(cv_datasets, test_size=0.3, random_state=7)

        file = open('error.txt', 'w')

        db = get_spacy_doc(file, train)
        db.to_disk('train_data.spacy')

        db = get_spacy_doc(file, test)
        db.to_disk('test_data.spacy')

        file.close()
        # fill_config(Path("CV-Parsing-using-Spacy-3/training/config.cfg"), Path("CV-Parsing-using-Spacy-3/training/base_config.cfg"))
        s4 = boto3.resource(
            service_name='s3',
            region_name='ap-south-1',
            aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
            aws_secret_access_key='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')
        model_list = list(configurable_json['model_name'])
        for model in model_list:
            #import pdb; pdb.set_trace()
            timestr = time.strftime("%Y%m%d-%H%M%S")

            print(section, model)
            config = configparser.ConfigParser()
            config.optionxform = str
            path = "/home/ptgml/resume/CV-Parsing-using-Spacy-3/data/training/config.cfg"
            config.read(path)
            client = boto3.client(service_name='s3', region_name='ap-south-1', aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
                                  aws_secret_access_key='c+zQLMpoaLUMNqjF1t/pSHaXgwROSPcQsmXcUKjP')
            result = client.get_object(Bucket='fractionable', Key='actualdata/samplejsons/confi.json')
            content = result['Body'].read().decode()
            configurable_json = json.loads(content)
            #print(configurable_json)
            config.sections()
            config['training']['max_steps'] = str(configurable_json['max_steps'])
            config['training']['eval_frequency'] = str(configurable_json['eval_frequency'])
            config['nlp']['batch_size'] = str(configurable_json['batch_size'])
            config['training.optimizer']['@optimizers'] = str(configurable_json['optimizer'])
            config['training.optimizer.learn_rate']['initial_rate'] = str(configurable_json['inital_rate'])
            config['components.transformer.model']['name'] = model
            with open(path, 'w') as f:
                config.write(f)
            spacy_train(Path("/home/ptgml/resume/CV-Parsing-using-Spacy-3/data/training/config.cfg"),
                        Path(f"./latest_train/{section}_{timestr}"), use_gpu=0,
                        overrides={"paths.train": "./train_data.spacy", "paths.dev": "./test_data.spacy"})
            path_output = f"./latest_train/{section}_{timestr}"
            nlp = spacy.load(f"{path_output}/model-best/")
            metric = json.load(open(f"{path_output}/model-best/meta.json"))
            response_123 = client.list_objects_v2(
                Bucket="fractionable",
                Prefix=f"golden_dataset/{section}")
            golden = []
            for res_123 in response_123['Contents'][1:]:
                res_123 = client.get_object(Bucket='fractionable', Key=res_123['Key'])
                content = res_123['Body']
                jsonObject123 = json.loads(content.read())
                golden.append(jsonObject123)
            # import pdb; pdb.set_trace()
            print("length of golden dataset", len(golden))
            Docs = golden
            x, kappa_score = generate_confusion_matrix(docs=Docs)
            dataset = json.dumps(cv_datasets)
            client.put_object(Body=dataset, Bucket='fractionable', Key=f'datasets/{section}/dataset_{kappa_score}.json')
            with mlflow.start_run(run_name=f"{section}_{timestr}"):
                mlflow.set_tag('model_flavor', f'spacy{section}')
                mlflow.spacy.log_model(spacy_model=nlp, artifact_path='model')
                mlflow.log_metric('f1_score', metric['performance']['ents_f'])
                mlflow.log_metric('length', len(cv_datasets))
                mlflow.log_metric('recall', (metric['performance']['ents_r']))
                mlflow.log_metric('precision', (metric['performance']['ents_p']))
                mlflow.log_metric('kappa', kappa_score)
                for enty in metric['performance']['ents_per_type'].keys():
                    mlflow.log_metric(f'{enty}_f1_score', metric['performance']['ents_per_type'][enty]['f'])
                    mlflow.log_metric(f'{enty}_recall', metric['performance']['ents_per_type'][enty]['r'])
                    mlflow.log_metric(f'{enty}_precision', metric['performance']['ents_per_type'][enty]['p'])
                accuracy = (metric['performance']['ents_f'] + metric['performance']['ents_p'] + metric['performance'][
                    'ents_r']) / 3
                mlflow.log_metric('accuracy', accuracy)
                config = configparser.RawConfigParser()
                config.read(f'{path_output}/model-best/config.cfg')
                details_dict = dict(config.items('nlp'))
                mlflow.log_param('batch_size', details_dict['batch_size'])
                mlflow.log_param('section', section)
                mlflow.log_param('section_model', f"{section}_{model}")
                details_dict = dict(config.items('components.transformer.model'))
                mlflow.log_param('model_name', details_dict['name'])
                details_dict = dict(config.items('training'))
                mlflow.log_param('max_steps', details_dict['max_steps'])
                mlflow.log_param('eval_frequency', details_dict['eval_frequency'])
                details_dict = dict(config.items('training.batcher'))
                mlflow.log_param('size', details_dict['size'])
                details_dict = dict(config.items('training.optimizer'))
                mlflow.log_param('optimizers', details_dict['@optimizers'])
                details_dict = dict(config.items('training.optimizer.learn_rate'))
                mlflow.log_param('initial_rate', details_dict['initial_rate'])
                my_run_id = mlflow.active_run().info.run_id
                # print(my_run_id)
            # import pdb; pdb.set_trace()

            filename1 = f'{section}_{timestr}'
            s3_file = s3fs.S3FileSystem()
            local_path = "./latest_train/" + filename1
            s3_path = "fractionable/Model/outputs"
            s3_file.put(local_path, s3_path, recursive=True)
            if kappa_score >= highest_kappa:
                nlp = spacy.load(f"{local_path}" + "/model-best")
                outputname=f'{section}'
                # df = pd.read_csv(r"kappa.csv")
                # df.append({'A':f'{outputname}','B':kappa_score},ignore_index=True)
                config_1 = nlp.config
                bytes_data = nlp.to_bytes()
                #print(dict(config), "--------------")
                #     #import json
                with open(f"{section}.json", "w") as outfile:
                    json.dump(dict(config_1), outfile)


                s4.Bucket('fractionable').upload_file(Filename=f'{section}.json',
                                                      Key=f'final_serialized_output/{section}.json')
                # bytes_data = nlp.to_bytes()
                with open(f"{section}.txt", "wb") as binary_file:
                    binary_file.write(bytes_data)
                s4.Bucket('fractionable').upload_file(Filename=f'{section}.txt',
                                                      Key=f'final_serialized_output/{section}.txt')

            #to make changes in meta json
            ensamable_dict=load_data(r"/home/ptgml/local_machine/ensamble.json")
            for entity in ensamable_dict.keys():
                # print(entity)
                meta_data = json.load(open(f"{path_output}/model-best/meta.json"))
                config123 = configparser.ConfigParser()
                config123.optionxform = str
                path = f"{path_output}/model-best/config.cfg"
                config123.read(path)
                #print(config['components.transformer.model']['name'])
                if entity in meta_data['performance']['ents_per_type'].keys():
                    if meta_data['performance']['ents_per_type'][entity]['f'] > ensamable_dict[entity]['f1_score']:
                        ensamable_dict[entity]['f1_score'] = meta_data['performance']['ents_per_type'][entity]['f']
                        ensamable_dict[entity]['model'] = model
                        ensamable_dict[entity]['section'] = section
                jsonstr = json.dumps(ensamable_dict)
                with open('ensamble.json', 'w') as f:
                        f.write(jsonstr)
        cv_datasets = []
            # import pdb; pdb.set_trace()
            # df=pd.read_csv(r"kappa.csv")
            # for i, j in zip(df['file_name'], df['kappa_score'].astype(float)):
            #     #if y > j:
            #     nlp = spacy.load(f"{local_path}" + "/model-best")
            #     # outputname=f'{outputname}'
            #     # df.append({'A':f'{outputname}','B':y},ignore_index=True)
            #     config_1 = nlp.config
            #     bytes_data = nlp.to_bytes()
            #     #print(dict(config), "--------------")
            #     #     #import json
            #     with open(f"{section}.json", "w") as outfile:
            #         json.dump(dict(config_1), outfile)
            #
            #     s4.Bucket('fractionable').upload_file(Filename=f'{section}.json',
            #                                           Key=f'final_serialized_output/{section}.json')
            #     # bytes_data = nlp.to_bytes()
            #     with open(f"{section}.txt", "wb") as binary_file:
            #         binary_file.write(bytes_data)
            #     s4.Bucket('fractionable').upload_file(Filename=f'{section}.txt',
            #                                           Key=f'final_serialized_output/{section}.txt')
            #     break

    # foo = pd.DataFrame({'message': ['timestr']})
    # foo.to_csv('foo.csv')
    # s4 = boto3.resource(
    #     service_name='s3',
    #     region_name='ap-south-1',
    #     aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
    #     aws_secret_access_key='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')
    # # s4 = boto3.resource('s3')
    # s4.Bucket('fractionable').upload_file(Filename='foo.csv', Key='test_lamda/foo.csv')


        """
        foo = pd.DataFrame({'message': ['timestr']})
        foo.to_csv('foo.csv')
        s4 = boto3.resource(
            service_name='s3',
            region_name='ap-south-1',
            aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
            aws_secret_access_key='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')
        #s4 = boto3.resource('s3')
        s4.Bucket('fractionable').upload_file(Filename='foo.csv', Key='test_lamda/foo.csv')


        filename1=f'outputs{timestr}'
        s3_file = s3fs.S3FileSystem()
        local_path = "/home/ptgml/latest_train/"+filename1
        s3_path = "fractionable/Model/outputs"
        s3_file.put(local_path, s3_path, recursive=True)"""
    # ec2.stop_instances(InstanceIds=instances)

    else:
        print(f"since new annotations are not greater than {number}")
#cv_data = json.load(open(r"/home/ubuntu/resume/argilla.json"))

