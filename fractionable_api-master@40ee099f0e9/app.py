import platform
import math
import requests
import nltk
from scipy.sparse.csgraph import connected_components
import numpy as np
# from nltk.corpus import stopwords
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from mangum import Mangum
import uvicorn
import sys, fitz
import shutil
import spacy
import numpy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
import io
import docx
import os
import unicodedata
import re
import pickle
import boto3
import boto3.session
# from elasticsearch import Elasticsearch
# from geopy.geocoders import Nominatim
import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from operator import itemgetter
import time
import uuid
# import pgeocode
from botocore.exceptions import ClientError
from pathlib import Path
import threading
import fasttext
import pandas as pd
nltk.data.path.append("/tmp")
nltk.download('stopwords', download_dir="/tmp")
nltk.download("punkt", download_dir="/tmp")
nltk.download('averaged_perceptron_tagger', download_dir="/tmp")
nltk.download('universal_tagset', download_dir="/tmp")
nltk.download('wordnet', download_dir="/tmp")
nltk.download('brown', download_dir="/tmp")
nltk.download('maxent_ne_chunker', download_dir="/tmp")

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()
handler = Mangum(app)

dic1 = {}


def is_pua(c):
    return unicodedata.category(c) == 'Co'


def my_replace(match):
    return str(2000 + int(match.group().strip()))


def graph_approach(df):
    start = df['start_date'].values
    end = df['end_date'].values
    skills = df['skill'].values
    graph = (start <= end[:, None]) & (end >= start[:, None]) & (skills == skills[:, None])
    n_components, indices = connected_components(graph, directed=False)
    return df.groupby(indices).aggregate({'skill': 'first', 'start_date': 'min', 'end_date': 'max'})


def download_ner_model(client):
    client.download_file(
        Bucket='fractionable',
        Key='final_serialized_output/entire_resume.txt',
        Filename="/tmp/entire_resume.txt"
    )
    client.download_file(
        Bucket='fractionable',
        Key='final_serialized_output/entire_resume.json',
        Filename="/tmp/entire_resume.json"
    )
    return 'success'


def download_skill_ner_model(client):
    client.download_file(
        Bucket='fractionable',
        Key='Model/models/skill_details.txt',
        Filename="/tmp/skill_details.txt"
    )
    client.download_file(
        Bucket='fractionable',
        Key='Model/models/skill_details.json',
        Filename="/tmp/skill_details.json"
    )
    return "success"


def get_phone_numbers(string):
    # r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    # import pdb; pdb.set_trace();
    us_phone_pattern = r'(\([2-9][0-9]{2}\)|[2-9][0-9]{2})[-.\s]([2-9][0-9]{2})[-.\s]([0-9]{4}\s)'
    indian_phone_pattern = r'[6789]\d{4}[-.\s]?\d{5}[.]?\s'
    us_phone_pattern_1 = r'(\([2-9][0-9]{2}\)|[2-9][0-9]{2})[-.\s]?([2-9][0-9]{2})[-.\s]?([0-9]{4}\s)'
    r_us = re.compile(us_phone_pattern)
    phone_numbers = r_us.findall(string + " ")
    if len(phone_numbers) == 0:
        r_in = re.compile(indian_phone_pattern)
        phone_numbers = r_in.findall(string + " ")
        if len(phone_numbers) == 0:
            r_us_1 = re.compile(us_phone_pattern_1)
            phone_numbers = r_us_1.findall(string + " ")
            if len(phone_numbers) == 0:
                phone_numbers = []
                country_flag = "IN"
            else:
                phone_numbers = [("-".join(x)).strip() for x in phone_numbers]
                country_flag = "US"
        else:
            phone_numbers = [("".join(x)).strip() for x in phone_numbers]
            country_flag = "IN"
    else:
        phone_numbers = [("-".join(x)).strip() for x in phone_numbers]
        country_flag = "US"
    return phone_numbers,country_flag


def get_email_addresses(string):
    # r = re.compile(r'[\w\.-]+@[\w\.-]+')
    r = re.compile(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[A-Z|a-z]{2,}\b')
    return r.findall(string)


def get_pincode(string):
    r = re.compile(r'\b\d{5,6}(?:-\d{4})?\b')
    pin = r.search(string)
    if pin != None:
        pin = pin.group()
    else:
        pin = ""
    return pin


def get_github_addresses(string):
    # r = re.compile(r'(https?://[^\s]+github\.com\S+)')
    # import pdb; pdb.set_trace();
    r = re.compile(r"(https://)(www\.)?(github.com/)([a-zA-Z0-9_-]+/)([a-zA-Z0-9_-]+)")
    matches = r.findall(string)
    if len(matches) > 0:
        matches = ["".join(x) for x in matches]
    else:
        r = re.compile(r"(http://)(www\.)?(github.com/)([a-zA-Z0-9_-]+/)([a-zA-Z0-9_-]+)")
        matches = r.findall(string)
        if len(matches) > 0:
            matches = ["".join(x) for x in matches]
        else:
            matches = []
    return matches


def get_linkInd_addresses(string):
    # import pdb; pdb.set_trace();
    # r = re.compile(r'(https?://[^\s]+linkedin\.com\S+)')
    r = re.compile(r"(https://)?(www\.)?(linkedin.com/in/)([a-zA-Z0-9_-]+)")
    matches = r.findall(string)
    if len(matches) > 0:
        matches = ["".join(x) for x in matches]
    else:
        r = re.compile(r"(http://)?(www\.)?(linkedin.com/in/)([a-zA-Z0-9_-]+)")
        matches = r.findall(string)
        if len(matches) > 0:
            matches = ["".join(x) for x in matches]
        else:
            matches = []
    return matches


def clean_text(txt):
    text = "".join([char for char in txt if not is_pua(char)])
    text = text.replace("–", "-")
    text = re.sub(r'[^\w@+_).//,/:(-]', " ", text)
    text = " ".join(text.split())
    text = text.strip()
    return text

def clean_text_(txt):
    text = "".join([char for char in txt if not is_pua(char)])
    text = text.replace("–", "-")
    text = re.sub(r'[^\w@+_).//,/:(-]', " ", text)
    text = " ".join(text.split())
    text = text.strip()
    return text

def tokenization(txt):
  lines=[l.strip() for l in txt.split('\n') if len(l)>0]
  lines=[nltk.word_tokenize(l) for l in lines]
  return lines;

def project_index_position(data):
    text_ = ""
    position = 0
    for l in data:
        text_ = text_ + str(clean_text_(" ".join(l))) + " "
        if (clean_text_(" ".join(l))).lower().strip() == "projects":
            position = len(text_) + 7
            break
    return position

def replace_function(string):
    string = string.replace(":", "")
    string = string.replace(".", "")
    string = string.replace(",", "")
    string = string.replace("&", "and")
    string = string.replace("-", "")
    string = string.replace('/', "")
    string = string.lower()
    return string

def date_extractor(ref_exp, flag, code):
    # import pdb; pdb.set_trace();
    ref_exp_duration = []
    if flag == "IN" and code == "IN":
        date_format_indicator = True
    elif flag == "US" and code == "US":
        date_format_indicator = False
    else:
        date_format_indicator = True
    for exp in ref_exp:
        Working_Date = exp['working_date']
        d1 = None
        d2 = None

        if "present" in Working_Date.lower() or "till now" in Working_Date.lower() or "tillnow" in Working_Date.lower() or "till" in Working_Date.lower() or "current" in Working_Date.lower() or "tilldate" in Working_Date.lower() or "till date" in Working_Date.lower():
            current_date = datetime.datetime.today().strftime('%Y/%m/%d')
            d1 = current_date
            d2 = Working_Date.lower()
        elif " to " in Working_Date.lower():
            try:
                d1, d2 = Working_Date.lower().split(" to ")
            except:
                pass
        elif "-" in Working_Date.lower():
            try:
                d1, d2 = Working_Date.lower().split("-")
            except:
                try:
                    d1, d2 = Working_Date.lower().split(" ")
                except:
                    pass
        else:
            try:
                d1, d2 = Working_Date.lower().split(" ")
            except:
                pass
        if isinstance(d1, type(None)) == False and isinstance(d2, type(None)) == False:
            d1 = d1.lower().strip()
            d2 = d2.lower().strip()
            # match = re.search('(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-zA-Z0-9.,-]*', d1)
            # if isinstance(match, type(None)) == False:
            #     try:
            #         d1 = re.sub(r'[^\w]', ' ', d1)
            #         d1 = parse(d1, dayfirst=True, fuzzy=True)
            #     except:
            #         d1 = None
            try:
                d1 = re.sub(r'[^\w]', ' ', d1)
                d1.strip()
                Match_1 = re.search("^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\s\d\d$", d1)
                Match_2 = re.search("^\d\d\s\d\d$", d1)
                if Match_1 != None:
                    d1 = "01" + " " + d1
                elif Match_2 != None:
                    d1 = "01" + " " + d1
                else:
                    pass
                try:
                    d1 = parse(d1, dayfirst=date_format_indicator, fuzzy=True)
                except:
                    d1 = ""
            except:
                d1 = ""
            # if "present" in d2 or "till now" in d2 or "tillnow" in d2 or "till" in d2 or "current" in d2 or "tilldate" in d2 or "till date" in d2:
            #     current_date = datetime.datetime.today().strftime('%Y/%m/%d')
            #     d2 = d2.lower()
            #     d2 = d2.replace("present", current_date)
            #     d2 = d2.replace("till now", current_date)
            #     d2 = d2.replace("tillnow", current_date)
            #     d2 = d2.replace("current", current_date)
            #     d2 = d2.replace("tilldate", current_date)
            #     d2 = d2.replace("till date", current_date)
            #     try:
            #         d2 = parse(d2, dayfirst=True, fuzzy=True)
            #     except:
            #         d2 = ""
            # else:
            try:
                d2 = re.sub(r'[^\w]', ' ', d2)
                d2 = d2.strip()
                Match_1 = re.search("^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\s\d\d$", d2)
                Match_2 = re.search("^\d\d\s\d\d$", d2 + " ")
                if Match_1 != None:
                    d2 = "01" + " " + d2
                elif Match_2 != None:
                    d2 = "01" + " " + d2
                else:
                    pass
                try:
                    d2 = parse(d2, dayfirst=date_format_indicator, fuzzy=True)
                except:
                    d2 = ""
            except:
                d2 = ""
            if d1 != "" and d2 != "":
                if d1 < d2:
                    temp_start = d1
                    temp_end = d2
                else:
                    temp_start = d2
                    temp_end = d1
                years = abs(relativedelta(temp_end, temp_start).years)
                months = abs(relativedelta(temp_end, temp_start).months)
                if years != 0:
                    temp_duration = str(years) + " " + "years" + " " + str(months) + " " + "months"
                else:
                    temp_duration = str(months) + " " + "months"
                temp_start = temp_start.strftime("%d/%m/%Y")
                temp_end = temp_end.strftime("%d/%m/%Y")
                temp_duration = temp_duration
                temp_year = years
                temp_months = months
            else:
                temp_start = ""
                temp_end = ""
                temp_duration = ""
                temp_year = ""
                temp_months =""
        else:
            temp_start = ""
            temp_end = ""
            temp_duration = ""
            temp_year = ""
            temp_months = ""
        if exp['company'] != "" or exp['designation'] != "":
            temp = {
                "working_date": exp['working_date'],
                "start_date": temp_start,
                "end_date": temp_end,
                "duration": temp_duration,
                "duration_in_years":temp_year,
                "duration_in_months": temp_months,
                "company": exp['company'],
                "company_loc": exp['company_loc'],
                "designation": exp['designation'],
                "spans_start": exp['spans_start'],
                "spans_end": exp['spans_end']
            }
            ref_exp_duration.append(temp)
    ref_exp_duration = sorted(ref_exp_duration, key=lambda k: k['spans_start'], reverse=False)
    return ref_exp_duration
def skill_age_extractor(skill_infos,ref_exp,skills,len_txt):
    skill_age_dict = []
    refined_skill_age = []
    for ind,exp in enumerate(ref_exp):
        for skill_info in skill_infos:
            temp_duration_start = exp["spans_end"]
            if ind + 1 <= len(ref_exp) - 1:
                temp_duration_end = ref_exp[ind + 1]['spans_start']
            else:
                temp_duration_end = len_txt
            if skill_info[0] in range(temp_duration_start,temp_duration_end) or skill_info[1] in range(temp_duration_start,temp_duration_end):
                # import pdb; pdb.set_trace();
                try:
                    temp_skill = {
                        "skill": skill_info[-1],
                        "start_date": datetime.datetime.strptime(exp['start_date'], '%d/%m/%Y'),
                        "end_date": datetime.datetime.strptime(exp['end_date'], '%d/%m/%Y')
                    }
                    skill_age_dict.append(temp_skill)
                except:
                    pass

    if len(skill_age_dict) > 0:
        df_skill = pd.DataFrame(skill_age_dict)
        df_skill = graph_approach(df_skill)
        df_skill['age'] = abs((df_skill['end_date']-df_skill['start_date'])/datetime.timedelta(days=365))
        skill_age_dict = df_skill.to_dict("records")
    else:
        skill_age_dict = []
    for skill in skills:
        temp = {}
        temp['skill'] = skill
        temp['age'] = 0
        for skill_age in skill_age_dict:
            if skill_age['skill'].lower() == skill.lower():
                temp['age'] = round(temp['age'] + float(skill_age['age']), 1)
        refined_skill_age.append(temp)
    return refined_skill_age

def create_split_resume_dic(txt: str):
    lines = txt.split('\n')
    df = pd.DataFrame()
    words = []
    len_words = []
    ln_startspan = []
    ln_endspan = []
    txt_line = []
    start = 0
    # check_file = Path("/tmp/nltk_data")
    # if check_file.is_file() == False:
    #     nltk.data.path.append("/tmp")
    #     nltk.download('stopwords', download_dir="/tmp")
    #     nltk.download("punkt", download_dir="/tmp")
    #     nltk.download('averaged_perceptron_tagger', download_dir="/tmp")
    #     nltk.download('universal_tagset', download_dir="/tmp")
    #     nltk.download('wordnet', download_dir="/tmp")
    #     nltk.download('brown', download_dir="/tmp")
    #     nltk.download('maxent_ne_chunker', download_dir="/tmp")
    for line in lines:
        if line != ' ':
            words.append(nltk.word_tokenize(line))
            len_words.append(len(nltk.word_tokenize(line)))
            ln_startspan.append(start)
            end = start + len(line)
            ln_endspan.append(end)
            txt_line.append(line)
            start = end + 1

    df['WORDS'] = words
    df['LEN_WORDS'] = len_words
    df['LN_STARTSPAN'] = ln_startspan
    df['LN_ENDSPAN'] = ln_endspan
    df['TXT_LINE'] = txt_line
    df = df[(df['LEN_WORDS'] > 0) & (df['LEN_WORDS'] <= 4)]
    df['TXT_LINE'] = df['TXT_LINE'].apply(lambda x: clean_text(x))
    df['TXT_LINE'] = df['TXT_LINE'].apply(lambda x: replace_function(x))
    df = df[df["TXT_LINE"] != "null"]
    fasttext_model = fasttext.load_model('utilities/model_v5_02_90.ftz')
    df['label'] = df['TXT_LINE'].apply(lambda x: fasttext_model.predict(x)[0][0].replace("__label__", ''))
    # import pdb; pdb.set_trace()
    df = df[df["label"] != "other"]
    df = df[df["label"] != "projects"]
    df.reset_index(inplace=True, drop=True)
    split_resume_dic = {}
    for index, row in df.iterrows():
        start = int(df.iloc[index]['LN_ENDSPAN'])
        if index + 1 < len(df) - 1:
            end = int(df.iloc[index + 1]['LN_STARTSPAN'])
        else:
            end = len(txt)
        if index == 0:
            start = 0
        if row['label'] not in split_resume_dic.keys():
            temp_txt = clean_text(txt[start:end])
            split_resume_dic[row['label']] = temp_txt
        else:
            tmp_txt = clean_text(txt[start:end])
            split_resume_dic[row['label']] += tmp_txt

    resume_txt = ""
    experience_starts = []
    experience_ends = []
    project_starts = []
    project_ends = []
    start = 0
    for key in split_resume_dic.keys():
        resume_txt += split_resume_dic[key]
        end = len(resume_txt)
        if key in ['work_experience', 'projects']:
            experience_starts.append(start)
            experience_ends.append(end)
        if key in ['projects']:
            project_starts.append(start)
            project_ends.append(end)
        start = len(resume_txt)
    split_resume_dic['resume_text'] = resume_txt
    if len(experience_starts) > 0 and len(experience_ends) > 0:
        split_resume_dic['experience_range_start'] = min(experience_starts)
        split_resume_dic['experience_range_ends'] = max(experience_ends)
    else:
        split_resume_dic['experience_range_start'] = 0
        split_resume_dic['experience_range_ends'] = 0
    if len(project_starts) > 0 and len(project_ends):
        split_resume_dic['project_range_starts'] = min(project_starts)
        split_resume_dic['project_range_ends'] = max(project_ends)
    else:
        split_resume_dic['project_range_starts'] = 0
        split_resume_dic['project_range_ends'] = 0
    print(df)
    return split_resume_dic

def entity_position_extractor(working_dates_info, company_info, company_loc_info, designation_info):
    range_dictionary = {}
    working_date_end_indices = [x[1] for x in working_dates_info]
    working_date_end_indices_sort = sorted(working_date_end_indices, reverse=False)
    for index, info in enumerate([company_info, company_loc_info, designation_info]):
        if index == 0:
            entity_info = "company_info"
        elif index == 1:
            entity_info = "company_loc_info"
        else:
            entity_info = "designation_info"
        if len(info) > 0:
            info = [x[1] for x in info]
            info = sorted(info, reverse=True)
            overlapping_info_in_positive_range = []
            overlapping_info_in_negative_range = []
            for end_index in working_date_end_indices_sort:
                overlapping_info_in_positive_range.extend([end for end in info if end in range(end_index, end_index+200)])
                overlapping_info_in_negative_range.extend([end for end in info if end in range(end_index-200, end_index)])
            overlapping_info_in_positive_range = list(set(overlapping_info_in_positive_range))
            overlapping_info_in_negative_range = list(set(overlapping_info_in_negative_range))
            # overlapping_info_in_positive_range.extend([end for end in info if end in range(working_date_end_indices_sort[-1], (working_date_end_indices_sort[-1]+200))])
            # overlapping_info_in_negative_range.extend([end for end in info if end in range((working_date_end_indices_sort[-1]-200), working_date_end_indices_sort[-1])])
            if len(overlapping_info_in_positive_range) > len(overlapping_info_in_negative_range):
                range_dictionary[entity_info] = 200
            elif len(overlapping_info_in_positive_range) < len(overlapping_info_in_negative_range):
                range_dictionary[entity_info] = -200
            elif len(overlapping_info_in_positive_range) == 0 and len(overlapping_info_in_negative_range) == 0:
                range_dictionary[entity_info] = 0
            else:
                range_dictionary[entity_info] = -200
        else:
            range_dictionary[entity_info] = 0
    return range_dictionary


def extract_predictions(model, txt, experience_range_start, experience_range_ends,project_index):
    if project_index == 0:
        project_index = len(txt)
    global dic1
    Ents = []
    name = []
    email = []
    phoneno = []
    person_address = []
    github_url = []
    linkedin_url = []
    summary = []
    totalexp = []
    company = []
    company_loc = []
    designation = []
    working_dates = []
    projects = []
    skills = []
    skills_infos = []
    high_education = []
    educational_institution = []
    passoutyear = []
    certifications = []
    awards = []
    working_dates_info = []
    company_info = []
    company_loc_info = []
    designation_info = []
    in_file = open(f"/tmp/{model}.txt", "rb")  # opening for [r]eading as [b]inary
    model_1 = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
    in_file.close()
    lang = "en"
    lang_cls = spacy.util.get_lang_class(lang)
    in_file = open(f"/tmp/{model}.json", "rb")  # opening for [r]eading as [b]inary
    json_1 = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
    in_file.close()
    nlp = lang_cls.from_config(json.loads(json_1))
    nlp.from_bytes(bytes(model_1))
    doc = nlp(txt)
    for ent in doc.ents:
        Ents.append([ent.label_, ent.start_char, ent.end_char])
        if ent.label_ == "NAME":
            name.append(ent.text)
        elif ent.label_ == "EMAIL":
            email.append(ent.text)
        elif ent.label_ == "PHONE_NO":
            phoneno.append(ent.text)
        elif ent.label_ == "PERSON_ADDRESS":
            person_address.append(ent.text)
        elif ent.label_ == "GITHUB_URL":
            github_url.append(ent.text)
        elif ent.label_ == "LINKEDIN_URL":
            linkedin_url.append(ent.text)
        elif ent.label_ == "SUMMARY":
            summary.append(ent.text)
        elif ent.label_ == "TOTAL_EXP":
            totalexp.append(ent.text)
        elif ent.label_ == "COMPANY" and ent.start_char < project_index:
            company.append(ent.text)
            company_info.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "COMPANY_LOC" and ent.start_char < project_index:
            company_loc.append(ent.text)
            company_loc_info.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "DESGINATION" and ent.start_char < project_index:
            designation.append(ent.text)
            designation_info.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "WORKING_DATES" and ent.start_char < project_index:
            working_dates.append(ent.text)
            working_dates_info.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "PROJECTS":
            projects.append(ent.text)
        elif ent.label_ == "SKILLS":
            skills.append(ent.text)
            skills_infos.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "HIGHER_EDUCATION":
            high_education.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "EDUCATIONAL_INSTITUTION":
            educational_institution.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "PASSOUTYEAR":
            passoutyear.append([ent.start_char, ent.end_char, ent.label_, ent.text])
        elif ent.label_ == "CERTIFICATIONS":
            certifications.append(ent.text)
        elif ent.label_ == "AWARDS":
            awards.append(ent.text)

    if len(name) > 0:
        name = list(set(name))
        name = [(x, len(x)) for x in name]
        name = sorted(name, key=itemgetter(1), reverse=True)[0][0]
    else:
        name = ""
    # import pdb; pdb.set_trace();
    if len(phoneno) > 0:
        phoneno_text = " ".join(phoneno)
        phoneno, country_flag = get_phone_numbers(phoneno_text)
        if len(phoneno) > 0:
            phoneno = phoneno[0].strip()
        else:
            phoneno, country_flag = get_phone_numbers(txt)
            if len(phoneno) > 0:
                phoneno = phoneno[0].strip()
            else:
                phoneno = ""

    else:
        phoneno, country_flag = get_phone_numbers(txt)
        if len(phoneno) > 0:
            phoneno = phoneno[0].strip()
        else:
            phoneno = ""

    if len(email) > 0:
        email_text = " ".join(email)
        email = get_email_addresses(email_text)
        if len(email) > 0:
            email = email[0]
        else:
            email = get_email_addresses(txt)
            if len(email) > 0:
                email = email[0]
            else:
                email = ""
    else:
        email = get_email_addresses(txt)
        if len(email) > 0:
            email = email[0]
        else:
            email = ""

    if len(person_address) > 0:
        person_address = person_address[0]
    else:
        person_address = ""
    extracted_pincode_TXT = get_pincode(txt)
    if extracted_pincode_TXT != "":
        pincode_ = extracted_pincode_TXT
    else:
        pincode_ = ""
    if len(str(pincode_)) == 6 and pincode_ != "":
        pincode_data_IN = pd.read_csv('utilities/city_pincode_IN.csv', encoding='iso-8859-1')
        row = pincode_data_IN[pincode_data_IN['pincode'] == int(pincode_)]
        if not row.empty:
            if isinstance(row['divisionname'].values[0], str):
                city = row['divisionname'].values[0]
            else:
                city = ""
            if isinstance(row['statename'].values[0], str):
                state = row['statename'].values[0]
            else:
                state = ""
            country = 'India'
            country_code = 'IN'
            pincode = pincode_
        else:
            city = ""
            state = ""
            pincode = ""
            country = ""
            country_code = ""
    elif pincode_ != "":
        pincode_data_US = pd.read_excel(r"utilities/city_pincode_US.xlsx", sheet_name="Zip Code Data")
        row = pincode_data_US[pincode_data_US['zip'] <= int(pincode_)]
        if not row.empty:
            if isinstance(row['primary_city'].values[-1], str):
                city = row['primary_city'].values[-1]
            else:
                city = ""
            if isinstance(row['state'].values[-1], str):
                state = row['state'].values[-1]
            else:
                state = ""

            country = 'USA'
            country_code = 'US'
            pincode = pincode_
        else:
            city = ""
            state = ""
            pincode = ""
            country = ""
            country_code = ""
    else:
        city = ""
        state = ""
        pincode = ""
        country = ""
        country_code = ""
    # import pdb; pdb.set_trace();
    github_url = get_github_addresses(txt)
    if len(github_url) > 0:
        github_url = github_url[0]
    else:
        github_url = ""

    linkedin_url = get_linkInd_addresses(txt)
    if len(linkedin_url) > 0:
        linkedin_url = linkedin_url[0]
    else:
        linkedin_url = ""
    if len(summary) > 0:
        summary = list(set(summary))[0]
    else:
        summary = ""
    # import pdb; pdb.set_trace();
    if len(totalexp) > 0:
        totalexp = list(set(totalexp))[0]
        totalexp = (totalexp.lower()).replace("+", "")
        match = re.search(r'(\d{1,2})(\s?)(years|year)', totalexp)
        if isinstance(match, type(None)) == False:
            exp_year = match.group()
            try:
                exp_year = int(exp_year.split(" ")[0])
            except:
                exp_year = ""
        else:
            exp_year = ""
        match = re.search(r'(\d{1,2})(\s?)(months|month)', totalexp)
        if isinstance(match, type(None)) == False:
            exp_month = match.group()
            try:
                exp_month = int(exp_month.split(" ")[0])
                if exp_year == "":
                    exp_year = 0
            except:
                exp_month = ""
        else:
            exp_month = ""
        if exp_year != "" and exp_month == "":
            exp_month = 0
        totalexp = {
            "year": exp_year,
            "month": exp_month
        }
    else:
        totalexp = {
            "year": "",
            "month": ""
        }
    company = list(set(company))
    company_loc = list(set(company_loc))
    designation = list(set(designation))
    working_dates = list(set(working_dates))
    projects = list(set(projects))
    skills = [skill.lower() for skill in skills]
    skills = list(set(skills))
    if len(high_education) > 0:
        high_education = sorted(high_education, key=itemgetter(0), reverse=False)
        high_education = [x[3] for x in high_education]
    if len(educational_institution) > 0:
        educational_institution = sorted(educational_institution, key=itemgetter(0), reverse=False)
        educational_institution = [x[3] for x in educational_institution]
    if len(passoutyear) > 0:
        passoutyear = sorted(passoutyear, key=itemgetter(0), reverse=False)
        passoutyear = [x[3] for x in passoutyear]
    certifications = list(set(certifications))
    awards = list(set(awards))
    refined_experience_details = []
    refined_skill_age = []
    if len(working_dates_info) > 0 and (len(company_info) > 0 or len(company_loc_info) > 0 or len(designation_info) > 0):
        # import pdb; pdb.set_trace();
        if len(working_dates_info) == 1:
            min_working_dates_end_index_diff = 300
        else:
            min_working_dates_end_index_diff = min(np.diff([x[1] for x in working_dates_info]))
        if min_working_dates_end_index_diff <= 250:
            position_info_dict = entity_position_extractor(working_dates_info=working_dates_info,
                                                      company_info=company_info,
                                                      company_loc_info=company_loc_info,
                                                      designation_info=designation_info)
            # import pdb; pdb.set_trace();
            if position_info_dict['company_info'] > 0:
                company_info = sorted(company_info, key=itemgetter(0), reverse=False)
            elif position_info_dict['company_info'] < 0:
                company_info = sorted(company_info, key=itemgetter(1), reverse=True)
            else:
                pass
            if position_info_dict['company_loc_info'] > 0:
                company_loc_info = sorted(company_loc_info, key=itemgetter(0), reverse=False)
            elif position_info_dict['company_loc_info'] < 0:
                company_loc_info = sorted(company_loc_info, key=itemgetter(1), reverse=True)
            else:
                pass
            if position_info_dict['designation_info'] > 0:
                designation_info = sorted(designation_info, key=itemgetter(0), reverse=False)
            elif position_info_dict['designation_info'] < 0:
                designation_info = sorted(designation_info, key=itemgetter(1), reverse=True)
            else:
                pass
            for ind, date in enumerate(working_dates_info):
                temp_date = date[-1].lower()
                if position_info_dict['company_info'] != 0:
                    temp_company = [x[-1] for x in company_info if x[1] in range(min(date[1], (date[1] + position_info_dict['company_info'])), max(date[1], (date[1] + position_info_dict['company_info'])))]
                    temp_company = temp_company[0] if len(temp_company) > 0 else ""
                else:
                    temp_company = ""
                if position_info_dict['company_loc_info'] != 0:
                    temp_company_loc = [x[-1] for x in company_loc_info if x[1] in range(min(date[1], (date[1] + position_info_dict['company_loc_info'])), max(date[1], (date[1] + position_info_dict['company_loc_info'])))]
                    temp_company_loc = temp_company_loc[0] if len(temp_company_loc) > 0 else ""
                else:
                    temp_company_loc = ""
                if position_info_dict['designation_info'] != 0:
                    temp_designation = [x[-1] for x in designation_info if x[1] in range(min(date[1], (date[1] + position_info_dict['designation_info'])), max(date[1], (date[1] + position_info_dict['designation_info'])))]
                    temp_designation = temp_designation[0] if len(temp_designation) > 0 else ""
                else:
                    temp_designation = ""

                temp = {
                    "working_date": temp_date,
                    "company": temp_company,
                    "company_loc": temp_company_loc,
                    "designation": temp_designation,
                    "spans_start": date[0],
                    "spans_end": date[1]
                }
                refined_experience_details.append(temp)
        else:
            for ind, date in enumerate(working_dates_info):
                temp_date = date[-1].lower()
                if len(company_info) > 0:
                    # temp_company = [x[-1] for x in company_info if x[1] in range((date[0]-250), (date[0]+250))]
                    # temp_company = temp_company[0] if len(temp_company) > 0 else ""
                    company_info_neg = sorted(company_info, key=itemgetter(1), reverse=True)
                    company_info_pos = sorted(company_info, key=itemgetter(1), reverse=False)
                    temp_company_neg = [x[-1] for x in company_info_neg if x[1] in range((date[0] - 250), date[0])]
                    temp_company_pos = [x[-1] for x in company_info_pos if x[1] in range(date[0], date[0] + 250)]
                    if len(temp_company_neg) > 0:
                        temp_company = temp_company_neg[0]
                    elif len(temp_company_pos) > 0:
                        temp_company = temp_company_pos[0]
                    else:
                        temp_company = ""
                else:
                    temp_company = ""
                if len(company_loc_info) > 0:
                    # temp_company_loc = [x[-1] for x in company_loc_info if x[1] in range((date[0]-250), (date[0]+250))]
                    # temp_company_loc = temp_company_loc[0] if len(temp_company_loc) > 0 else ""
                    company_loc_info_neg = sorted(company_loc_info, key=itemgetter(1), reverse=True)
                    company_loc_info_pos = sorted(company_loc_info, key=itemgetter(1), reverse=False)
                    temp_company_loc_neg = [x[-1] for x in company_loc_info_neg if x[1] in range((date[0] - 250), date[0])]
                    temp_company_loc_pos = [x[-1] for x in company_loc_info_pos if x[1] in range(date[0], date[0] + 250)]
                    if len(temp_company_loc_neg) > 0:
                        temp_company_loc = temp_company_loc_neg[0]
                    elif len(temp_company_loc_pos) > 0:
                        temp_company_loc = temp_company_loc_pos[0]
                    else:
                        temp_company_loc = ""
                else:
                    temp_company_loc = ""
                if len(designation_info) > 0:
                    # temp_designation = [x[-1] for x in designation_info if x[1] in range((date[0]-250), (date[0]+250))]
                    # temp_designation = temp_designation[0] if len(temp_designation) > 0 else ""
                    designation_info_neg = sorted(designation_info, key=itemgetter(1), reverse=True)
                    designation_info_pos = sorted(designation_info, key=itemgetter(1), reverse=False)
                    temp_designation_neg = [x[-1] for x in designation_info_neg if x[1] in range((date[0] - 250), date[0])]
                    temp_designation_pos = [x[-1] for x in designation_info_pos if x[1] in range(date[0], date[0] + 250)]
                    if len(temp_designation_neg) > 0:
                        temp_designation = temp_designation_neg[0]
                    elif len(temp_designation_pos) > 0:
                        temp_designation = temp_designation_pos[0]
                    else:
                        temp_designation = ""
                else:
                    temp_designation = ""
                temp = {
                    "working_date": temp_date,
                    "company": temp_company,
                    "company_loc": temp_company_loc,
                    "designation": temp_designation,
                    "spans_start": date[0],
                    "spans_end": date[1]
                }
                refined_experience_details.append(temp)

        refined_experience_details = date_extractor(ref_exp=refined_experience_details, flag=country_flag, code=country_code)
        refined_skill_age = skill_age_extractor(skill_infos=skills_infos, ref_exp=refined_experience_details, skills=skills, len_txt=len(txt))
        refined_experience_details = sorted(refined_experience_details, key=lambda k: k['spans_start'], reverse=False)
    dictionary = {
        "Person_Details": {
            "name": name,
            "phoneno": phoneno,
            "email": email,
            "person_address": person_address,
            "city": city,
            "state": state,
            "pincode": pincode,
            "country": country,
            "country_code": country_code,
            "github_url": github_url,
            "linkedin_url": linkedin_url,
            "summary": summary,
            "totalexp": totalexp
        },
        "Experience_Details": {
            "company": company,
            "company_loc": company_loc,
            "designation": designation,
            "working_dates": working_dates,
            "projects": projects,
            "skills": skills,
        },
        "Education_Details": {
            "high_education": high_education,
            "educational_institution": educational_institution,
            "passoutyear": passoutyear,
            "certifications": certifications,
            "awards": awards,
        },
        "refined_experience_details": refined_experience_details,
        "refined_skill_age": refined_skill_age
    }
    dic1[model] = dictionary
    return "success"


def predict_skill_details(txt, model, address, country_code):
    try:
        global dic1
        city = dic1['entire_resume']['Person_Details']['city']
        state = dic1['entire_resume']['Person_Details']['state']
        country = dic1['entire_resume']['Person_Details']['country']
        country_code = dic1['entire_resume']['Person_Details']['country_code']
        if txt != "" and address != "" and country_code == "":
            Ents = []
            skills = []
            # in_file = open("model/my_file_skill.txt", "rb")  # opening for [r]eading as [b]inary
            in_file = open(f"/tmp/{model}.txt", "rb")
            model_1 = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
            in_file.close()
            lang = "en"
            lang_cls = spacy.util.get_lang_class(lang)
            # in_file = open(r"model/sample_test_skill_config.json")
            in_file = open(f"/tmp/{model}.json")
            json_1 = in_file.read()
            nlp = lang_cls.from_config(json.loads(json_1))
            nlp.from_bytes(bytes(model_1))
            nlp_city = lang_cls.from_config(json.loads(json_1))
            nlp_city.from_bytes(bytes(model_1))
            skills_from_json = r"utilities/jz_skill_patterns.jsonl"
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.from_disk(skills_from_json)
            cities_from_json = r"utilities/city_state.jsonl"
            cities_ruler = nlp_city.add_pipe("entity_ruler", before="ner")
            cities_ruler.from_disk(cities_from_json)
            doc = nlp(txt)
            for ent in doc.ents:
                Ents.append([ent.label_, ent.start_char, ent.end_char, ent.text])
                if ent.label_ == "SKILL":
                    skills.append(ent.text.lower())
            city_doc = nlp_city(address)
            states = []
            cities = []
            for ent in city_doc.ents:
                if ent.label_ == "STATE":
                    states.append(ent.text.lower())
                elif ent.label_ == "CITY":
                    cities.append(ent.text.lower())
            df_cities = pd.read_csv(r"utilities/list_of_cities_us.csv")
            df_cities['cities'] = df_cities['cities'].apply(lambda x: str(x).lower())
            df_cities['state'] = df_cities['state'].apply(lambda x: str(x).lower())
            if len(cities) > 0:
                city = cities[0]
                state = df_cities.loc[df_cities['cities'] == city, 'state'].iloc[0]
                country = df_cities.loc[df_cities['cities'] == city, 'country'].iloc[0]
                country_code = df_cities.loc[df_cities['cities'] == city, 'country_code'].iloc[0]
            elif len(states) > 0:
                state = states[0]
                city = df_cities.loc[df_cities['state'] == state, 'cities'].iloc[0]
                country = df_cities.loc[df_cities['state'] == state, 'country'].iloc[0]
                country_code = df_cities.loc[df_cities['state'] == state, 'country_code'].iloc[0]
            else:
                city = ""
                state = ""
                country = ""
                country_code = ""
            dictionary = {
                "SKILLS": skills
            }
        elif txt != "":
            Ents = []
            skills = []
            # in_file = open("model/my_file_skill.txt", "rb")  # opening for [r]eading as [b]inary
            in_file = open(f"/tmp/{model}.txt", "rb")
            model_1 = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
            in_file.close()
            lang = "en"
            lang_cls = spacy.util.get_lang_class(lang)
            # in_file = open(r"model/sample_test_skill_config.json")
            in_file = open(f"/tmp/{model}.json")
            json_1 = in_file.read()
            nlp = lang_cls.from_config(json.loads(json_1))
            nlp.from_bytes(bytes(model_1))
            skills_from_json = r"utilities/jz_skill_patterns.jsonl"
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.from_disk(skills_from_json)
            doc = nlp(txt)
            for ent in doc.ents:
                Ents.append([ent.label_, ent.start_char, ent.end_char, ent.text])
                if ent.label_ == "SKILL":
                    skills.append(ent.text.lower())
            dictionary = {
                "SKILLS": skills
            }
        elif address != "" and country_code == "":
            in_file = open(f"/tmp/{model}.txt", "rb")
            model_1 = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
            in_file.close()
            lang = "en"
            lang_cls = spacy.util.get_lang_class(lang)
            # in_file = open(r"model/sample_test_skill_config.json")
            in_file = open(f"/tmp/{model}.json")
            json_1 = in_file.read()
            nlp_city = lang_cls.from_config(json.loads(json_1))
            nlp_city.from_bytes(bytes(model_1))
            cities_from_json = r"utilities/city_state.jsonl"
            cities_ruler = nlp_city.add_pipe("entity_ruler", before="ner")
            cities_ruler.from_disk(cities_from_json)
            city_doc = nlp_city(address)
            states = []
            cities = []
            for ent in city_doc.ents:
                if ent.label_ == "STATE":
                    states.append(ent.text.lower())
                elif ent.label_ == "CITY":
                    cities.append(ent.text.lower())
            df_cities = pd.read_csv(r"utilities/list_of_cities_us.csv")
            df_cities['cities'] = df_cities['cities'].apply(lambda x: str(x).lower())
            df_cities['state'] = df_cities['state'].apply(lambda x: str(x).lower())
            if len(cities) > 0:
                city = cities[0]
                state = df_cities.loc[df_cities['cities'] == city, 'state'].iloc[0]
                country = df_cities.loc[df_cities['cities'] == city, 'country'].iloc[0]
                country_code = df_cities.loc[df_cities['cities'] == city, 'country_code'].iloc[0]
            elif len(states) > 0:
                state = states[0]
                city = df_cities.loc[df_cities['state'] == state, 'cities'].iloc[0]
                country = df_cities.loc[df_cities['state'] == state, 'country'].iloc[0]
                country_code = df_cities.loc[df_cities['state'] == state, 'country_code'].iloc[0]
            else:
                city = ""
                state = ""
                country = ""
                country_code = ""
            dictionary = {
                "SKILLS": []
            }
        else:
            dictionary = {
                "SKILLS": []
            }
            city = dic1['entire_resume']['Person_Details']['city']
            state = dic1['entire_resume']['Person_Details']['state']
            country = dic1['entire_resume']['Person_Details']['country']
            country_code = dic1['entire_resume']['Person_Details']['country_code']
        dic1[model] = dictionary
        if "entire_resume" in dic1.keys():
            dic1['entire_resume']['Person_Details']['city'] = city
            dic1['entire_resume']['Person_Details']['state'] = state
            dic1['entire_resume']['Person_Details']['country'] = country
            dic1['entire_resume']['Person_Details']['country_code'] = country_code

    except:
        sys.exit(f"unable to fetch skills from skills ner model")
    return "success"


def get_exp_details(txt: str):
    global dic1
    url = "https://4xyrvsf2k376rnap24j75yhbem0qmpwk.lambda-url.ap-south-1.on.aws/file"
    try:
        api_res = requests.post(url, txt=txt)
        api_res_json = api_res.json()
    except:
        api_res_json = {
            "Person_Details": {
                "name": "",
                "phoneno": "",
                "email": "",
                "person_address": "",
                "city": "",
                "state": "",
                "pincode": "",
                "country": "",
                "country_code": "",
                "github_url": "",
                "linkedin_url": "",
                "summary": "",
                "totalexp": {
                    "year" : "",
                    "month": ""
                }
            },
            "Experience_Details": {
                "company": [],
                "company_loc": [],
                "designation": [],
                "working_dates": [],
                "projects": [],
                "skills": [],
            },
            "Education_Details": {
                "high_education": [],
                "educational_institution": [],
                "passoutyear": [],
                "certifications": [],
                "awards": [],
            },
            "refined_experience_details": [],
            "refined_skill_age": []
        }
    dic1['experience_info'] = api_res_json
    return "success"


def get_education_details(txt: str):
    global dic1
    url = "https://xnqk4tn3f2ep7bjtemoyk7xwdu0rfjwl.lambda-url.ap-south-1.on.aws/file"
    try:
        api_res = requests.post(url, txt=txt)
        api_res_json = api_res.json()
    except:
        api_res_json = {
            "Person_Details": {
                "name": "",
                "phoneno": "",
                "email": "",
                "person_address": "",
                "city": "",
                "state": "",
                "pincode": "",
                "country": "",
                "country_code": "",
                "github_url": "",
                "linkedin_url": "",
                "summary": "",
                "totalexp": {
                    "year" : "",
                    "month": ""
                }
            },
            "Experience_Details": {
                "company": [],
                "company_loc": [],
                "designation": [],
                "working_dates": [],
                "projects": [],
                "skills": [],
            },
            "Education_Details": {
                "high_education": [],
                "educational_institution": [],
                "passoutyear": [],
                "certifications": [],
                "awards": [],
            },
            "refined_experience_details": [],
            "refined_skill_age": []
        }
    dic1['education_info'] = api_res_json
    return "success"


@app.post("/file")
async def upload_file(uploaded_Resume: UploadFile = File(...)):
    global dic1
    message = ''
    try:
        file_loc = f"/tmp/{uploaded_Resume.filename}"
        with open(file_loc, "wb+") as file_object:
            shutil.copyfileobj(uploaded_Resume.file, file_object)
        message = message + 'Resume Uploaded Successful'
    except:
        message = message + 'Resume Uploaded failed'
    fname = f"/tmp/{uploaded_Resume.filename}"
    doc = fitz.open(fname)
    text = ''
    profile = " ".join((fname.split(".")[0]).split("_")[0:2])
    page_no = 1
    # for page in doc:
    #     if page_no == 1:
    #         blocks = page.get_text("dict")["blocks"]
    #         if len(blocks) == 1:
    #             # sys.exit(f"{uploaded_Resume.filename} is a scanned pdf")
    #             final = {}
    #             final['label'] = {}
    #             final['label']['Person_Details'] = {
    #                 "name": "",
    #                 "phoneno": "",
    #                 "email": "",
    #                 "person_address": "",
    #                 "city": "",
    #                 "state": "",
    #                 "pincode": "",
    #                 "country": "",
    #                 "country_code": "",
    #                 "github_url": "",
    #                 "linkedin_url": "",
    #                 "summary": "",
    #                 "totalexp": {
    #                     "year": "",
    #                     "month": ""
    #                 }
    #             }
    #             final['label']['Education_Details'] = {
    #                 "high_education": [],
    #                 "educational_institution": [],
    #                 "passoutyear": [],
    #                 "certifications": [],
    #                 "awards": [],
    #             }
    #             final['label']['Experience_Details'] = {
    #                 "company": [],
    #                 "company_loc": [],
    #                 "designation": [],
    #                 "working_dates": [],
    #                 "projects": [],
    #                 "skills": []
    #             }
    #             final['label']['refined_experience_details'] = []
    #             final['label']['refined_skill_age'] = []
    #             final['label']['carrer_growth_score'] = ""
    #             final['label']['loyality_score'] = ""
    #             final['label']['message'] = f"{uploaded_Resume.filename} is a scanned pdf. Resume parser is not able to parse Scanned pdf"
    #             json_object = json.dumps(final, indent=4)
    #             final_json = json.loads(json_object)
    #             return final_json
    #     page_no = page_no + 1
    if profile.lower() == "dice profile":
        resume_name = " ".join((fname.split(".")[0]).split("_")[2:])
        page_no = 1
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            temp = {'size': blocks[0]["lines"][0]["spans"][0]['size'],
                    'flags': blocks[0]["lines"][0]["spans"][0]['flags'],
                    'font': blocks[0]["lines"][0]["spans"][0]['font'],
                    'color': blocks[0]["lines"][0]["spans"][0]['color'],
                    'text': blocks[0]["lines"][0]["spans"][0]['text']}
            page_no += 1
            if temp['size'] == 21.5 and temp['flags'] == 16 and temp['font'] == "Arial-BoldMT" and temp['color'] == 13369344 and temp['text'] == resume_name:
                break
            else:
                text = text + str(page.get_text())
    else:
        for page in doc:
            text = text + str(page.get_text())
    line_data = tokenization(text)
    project_index = project_index_position(line_data)
    resume_split = create_split_resume_dic(txt=text)
    text = clean_text(text)

    final = {}
    # import pdb; pdb.set_trace();
    if text.strip() != "":
        check_file = Path("/tmp/entire_resume.json")
        if check_file.is_file() == False:
            client = boto3.client(service_name='s3', region_name='ap-south-1', aws_access_key_id='AKIAZP5PPFU6K5W2OJXZ',
                                  aws_secret_access_key='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')
            t1 = threading.Thread(target=download_ner_model, args=(client,))
            t2 = threading.Thread(target=download_skill_ner_model, args=(client,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        skills_text = ""
        if 'skills' in resume_split.keys():
            skills_text += resume_split['skills']
        t3 = extract_predictions(model="entire_resume", txt=text,
                                 experience_range_start=resume_split['experience_range_start'],
                                 experience_range_ends=resume_split['experience_range_ends'],
                                 project_index=project_index)
        # import pdb; pdb.set_trace();
        person_address = dic1['entire_resume']['Person_Details']['person_address']
        person_address = person_address.lower()
        t4 = predict_skill_details(txt=skills_text, model="skill_details", address=person_address, country_code=dic1['entire_resume']['Person_Details']['country_code'])
        # if "work_experience" in resume_split.keys() and "education" in resume_split.keys():
        #     t5 = threading.Thread(target=get_exp_details, args=(resume_split['work_experience'],))
        #     t6 = threading.Thread(target=get_education_details, args=(resume_split['education'],))
        #     t5.start()
        #     t6.start()
        #     t5.join()
        #     t6.join()
        # elif "work_experience" in resume_split.keys():
        #     t5 = threading.Thread(target=get_exp_details, args=(resume_split['work_experience'],))
        #     t5.start()
        #     t5.join()
        # elif "education" in resume_split.keys():
        #     t6 = threading.Thread(target=get_education_details, args=(resume_split['education'],))
        #     t6.start()
        #     t6.join()
        # else:
        #     pass
        dic = dic1
        dic1 = {}
        final['label'] = {}
        if "entire_resume" in dic.keys():
            final['label']['Person_Details'] = dic['entire_resume']['Person_Details']
            final['label']['Education_Details'] = dic['entire_resume']['Education_Details']
            final['label']['Experience_Details'] = dic['entire_resume']['Experience_Details']
            final['label']['refined_experience_details'] = dic['entire_resume']['refined_experience_details']
            final['label']['refined_skill_age'] = dic['entire_resume']['refined_skill_age']
            if len(final['label']['refined_experience_details']) > 0:
                duration_in_year = []
                designation_growth = []
                # import pdb; pdb.set_trace();
                for exp in final['label']['refined_experience_details']:
                    try:
                        # duration = exp['duration']
                        years = exp['duration_in_years']
                        months = exp['duration_in_months']
                        duration_in_years = round((float(years) + (float(months))/12), 2)
                        duration_in_year.append(duration_in_years)
                        designation_growth.append(exp['designation'])
                    except:
                        pass
                # import pdb; pdb.set_trace();
                if len(duration_in_year) > 0 and sum(duration_in_year) > 0:
                    loyality = round((sum(duration_in_year) / len(duration_in_year)), 2)
                    designations_count_which_are_predicted = [des for des in designation_growth if des != ""]
                    designations_count_which_are_not_predicted = len(designation_growth) - len(designations_count_which_are_predicted)
                    designations_count_which_are_predicted_unique = len(list(set(designations_count_which_are_predicted)))
                    total_unique_designations = designations_count_which_are_predicted_unique + designations_count_which_are_not_predicted
                    carrer_growth_avg = sum(duration_in_year) / total_unique_designations
                    if carrer_growth_avg <= 2:
                        carrer_growth_score = 100
                    elif carrer_growth_avg >= 10:
                        carrer_growth_score = 1
                    else:
                        carrer_growth_score = round(((-carrer_growth_avg - (-10)) / (-2 - (-10))) * 100, 2)
                    print("carrer_growth_avg", carrer_growth_avg)
                    if final['label']['Person_Details']['totalexp']["year"] == "" and final['label']['Person_Details']['totalexp']['month'] == "":
                        total_exp = sum(duration_in_year)
                        years = int(total_exp)
                        months = int((total_exp*12) % 12)
                        final['label']['Person_Details']['totalexp']["year"] = years
                        final['label']['Person_Details']['totalexp']['month'] = months
                else:
                    carrer_growth_score = ""
                    loyality = ""

                final['label']['carrer_growth_score'] = carrer_growth_score
                final['label']['loyality_score'] = loyality
                final['label']['message'] = ""

            else:
                final['label']['carrer_growth_score'] = ""
                final['label']['loyality_score'] = ""
                final['label']['message'] = ""
        else:
            final['label']['Person_Details'] = {
                "name": "",
                "phoneno": "",
                "email": "",
                "person_address": "",
                "city": "",
                "state": "",
                "pincode": "",
                "country": "",
                "country_code": "",
                "github_url": "",
                "linkedin_url": "",
                "summary": "",
                "totalexp": {
                    "year" : "",
                    "month": ""
                }
            }
            final['label']['Education_Details'] = {
                "high_education": [],
                "educational_institution": [],
                "passoutyear": [],
                "certifications": [],
                "awards": [],
            }
            final['label']['Experience_Details'] = {
                "company": [],
                "company_loc": [],
                "designation": [],
                "working_dates": [],
                "projects": [],
                "skills": []
            }
            final['label']['refined_experience_details'] = []
            final['label']['refined_skill_age'] = []
            final['label']['carrer_growth_score'] = ""
            final['label']['loyality_score'] = ""
        # if len(dic['entire_resume']['refined_experience_details']) >= len(dic['experience_info']['refined_experience_details']):
        #     final['label']['refined_experience_details'] = dic['entire_resume']['refined_experience_details']
        #     final['label']['refined_skill_age'] = dic['entire_resume']['refined_skill_age']
        #     final['label']['Experience_Details'] = dic['entire_resume']['Experience_Details']
        # else:
        #     final['label']['refined_experience_details'] = dic['experience_info']['refined_experience_details']
        #     final['label']['refined_skill_age'] = dic['experience_info']['refined_skill_age']
        #     final['label']['Experience_Details'] = dic['experience_info']['Experience_Details']
        # if "skill_details" in dic.keys():
        #     final['label']['Experience_Details']['skills'] += dic['skill_details']['SKILLS']
        #     final['label']['Experience_Details']['skills'] = list(
        #         set(final['label']['Experience_Details']['skills']))

    else:
        # sys.exit(f"no content in {uploaded_Resume.filename}")
        final = {}
        final['label'] = {}
        final['label']['Person_Details'] = {
            "name": "",
            "phoneno": "",
            "email": "",
            "person_address": "",
            "city": "",
            "state": "",
            "pincode": "",
            "country": "",
            "country_code": "",
            "github_url": "",
            "linkedin_url": "",
            "summary": "",
            "totalexp": {
                "year": "",
                "month": ""
            }
        }
        final['label']['Education_Details'] = {
            "high_education": [],
            "educational_institution": [],
            "passoutyear": [],
            "certifications": [],
            "awards": [],
        }
        final['label']['Experience_Details'] = {
            "company": [],
            "company_loc": [],
            "designation": [],
            "working_dates": [],
            "projects": [],
            "skills": []
        }
        final['label']['refined_experience_details'] = []
        final['label']['refined_skill_age'] = []
        final['label']['carrer_growth_score'] = ""
        final['label']['loyality_score'] = ""
        final['label']['message'] = f"{uploaded_Resume.filename} is a scanned pdf. Resume parser is not able to parse Scanned pdf"
        json_object = json.dumps(final, indent=4)
        final_json = json.loads(json_object)
        return final_json
    json_object = json.dumps(final, indent=4)
    final_json = json.loads(json_object)
    return final_json


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

