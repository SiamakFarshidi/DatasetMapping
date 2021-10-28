import shlex
import json
import subprocess
import requests
import os
from urllib.parse import urlparse
from WebCrawler import Crawler
#from LanguageDetection import LangaugePrediction
from Synonyms import getSynonyms
from os import walk
from bs4 import BeautifulSoup
import requests
import json
import sys
import re
import spacy
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import enchant
from fuzzywuzzy import fuzz
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.request
import uuid
import json, xmljson
from lxml.etree import fromstring, tostring
import xml.etree.ElementTree as ET
import itertools
#----------------------------------------------------------------------------------------
#nltk.download('wordnet')
#nltk.download('stopwords')
#----------------------------------------------------------------------------------------
EnglishTerm = enchant.Dict("en_US")
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
Lda = gensim.models.ldamodel.LdaModel
spacy_nlp  = spacy.load('en_core_web_md')
#----------------------------------------------------------------------------------------
MetaDataRecordPath="./Metadata records/"
ICOS__MetadataRecordsFileName="ICOS-metadata-records.json"
SeaDataNet__MetadataRecordsFileName= "CDI-SeaDataNet-metadata-records.xml"
metadataStar_root="./Metadata*/metadata*.json"
RI_root="./Metadata*/RIs.json"
indexFiles_root="./index files/"
domain_root="./Metadata*/domain.json"
essentialVariabels_root="./Metadata*/essential_variables.json"
domainVocbularies_root="./Metadata*/Vocabularies.json"
#----------------------------------------------------------------------------------------
acceptedSimilarityThreshold=0.75
#----------------------------------------------------------------------------------------

#-------------------SeaDataNet
def getDatasetRecords__SeaDataNet_EDMED():

    with urllib.request.urlopen('https://edmed.seadatanet.org/sparql/sparql?query=select+%3FEDMEDRecord+%3FTitle+where+%7B%3FEDMEDRecord+a+%3Chttp%3A%2F%2Fwww.w3.org%2Fns%2Fdcat%23Dataset%3E+%3B+%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Fterms%2Ftitle%3E+%3FTitle+.%7D+&output=json&stylesheet=') as f:
        data = f.read().decode('utf-8')
    json_data = json.loads(data)
    indexFile= open(MetaDataRecordPath+SeaDataNet__MetadataRecordsFileName+"_EDMED","w+")
    indexFile.write(json.dumps(json_data))
    indexFile.close()
    print("SeaDataNet data collection is done!")
#-------------------
def getDatasetRecords__SeaDataNet_CDI():
    with urllib.request.urlopen('https://cdi.seadatanet.org/report/aggregation') as f:
        data = f.read().decode('utf-8')
    indexFile= open(MetaDataRecordPath+SeaDataNet__MetadataRecordsFileName,"w+")
    indexFile.write(data)
    indexFile.close()
    print("SeaDataNet data collection is done!")
#-------------------ICOS
def getDatasetRecords__ICOS():
    cURL = r"""curl https://meta.icos-cp.eu/sparql -X POST --data 'query=prefix%20cpmeta%3A%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fontologies%2Fcpmeta%2F%3E%0Aprefix%20prov%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2Fns%2Fprov%23%3E%0Aselect%20%3Fdobj%20%3Fspec%20%3FfileName%20%3Fsize%20%3FsubmTime%20%3FtimeStart%20%3FtimeEnd%0Awhere%20%7B%0A%09VALUES%20%3Fspec%20%7B%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FradonFluxSpatialL3%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2Fco2EmissionInventory%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FsunInducedFluorescence%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FoceanPco2CarbonFluxMaps%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FinversionModelingSpatial%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FbiosphereModelingSpatial%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FecoFluxesDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FecoEcoDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FecoMeteoDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FecoAirTempMultiLevelsDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FecoProfileMultiLevelsDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcMeteoL0DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcLosGatosL0DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcPicarroL0DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FingosInversionResult%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2Fsocat_DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcBioMeteoRawSeriesBin%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcStorageFluxRawSeriesBin%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcBioMeteoRawSeriesCsv%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcStorageFluxRawSeriesCsv%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcSaheatFlagFile%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FceptometerMeasurements%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FglobalCarbonBudget%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FnationalCarbonEmissions%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FglobalMethaneBudget%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FdigHemispherPics%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcEddyFluxRawSeriesCsv%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcEddyFluxRawSeriesBin%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcCh4L2DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcCoL2DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcCo2L2DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcMtoL2DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcC14L2DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcMeteoGrowingNrtDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcCo2NrtGrowingDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcCh4NrtGrowingDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcN2oL2DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcCoNrtGrowingDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcN2oNrtGrowingDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FingosCh4Release%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FingosN2oRelease%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatcRnNrtDataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2Fdrought2018AtmoProduct%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FmodelDataArchive%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcArchiveProduct%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2Fdought2018ArchiveProduct%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FatmoMeasResultsArchive%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcNrtAuxData%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcFluxnetProduct%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2Fdrought2018FluxnetProduct%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcNrtFluxes%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcNrtMeteosens%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FetcNrtMeteo%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FicosOtcL1Product%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FicosOtcL1Product_v2%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FicosOtcL2Product%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FicosOtcFosL2Product%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FotcL0DataObject%3E%20%3Chttp%3A%2F%2Fmeta.icos-cp.eu%2Fresources%2Fcpmeta%2FinversionModelingTimeseries%3E%7D%0A%09%3Fdobj%20cpmeta%3AhasObjectSpec%20%3Fspec%20.%0A%09%3Fdobj%20cpmeta%3AhasSizeInBytes%20%3Fsize%20.%0A%3Fdobj%20cpmeta%3AhasName%20%3FfileName%20.%0A%3Fdobj%20cpmeta%3AwasSubmittedBy%2Fprov%3AendedAtTime%20%3FsubmTime%20.%0A%3Fdobj%20cpmeta%3AhasStartTime%20%7C%20%28cpmeta%3AwasAcquiredBy%20%2F%20prov%3AstartedAtTime%29%20%3FtimeStart%20.%0A%3Fdobj%20cpmeta%3AhasEndTime%20%7C%20%28cpmeta%3AwasAcquiredBy%20%2F%20prov%3AendedAtTime%29%20%3FtimeEnd%20.%0A%09FILTER%20NOT%20EXISTS%20%7B%5B%5D%20cpmeta%3AisNextVersionOf%20%3Fdobj%7D%0A%7D%0Aorder%20by%20desc%28%3FsubmTime%29'"""
    lCmd = shlex.split(cURL) # Splits cURL into an array
    p = subprocess.Popen(lCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate() # Get the output and the err message
    json_data = json.loads(out.decode("utf-8"))
    indexFile= open(MetaDataRecordPath+ICOS__MetadataRecordsFileName,"w+")
    indexFile.write(json.dumps(json_data))
    indexFile.close()
    print("ICOS data collection is done!")
#----------------------------------------------------------------------------------------
def processDatasetRecords__ICOS(rnd,genRnd,startingPoint):
    indexFile= open(MetaDataRecordPath+ICOS__MetadataRecordsFileName,"r")
    dataset_json = json.loads(indexFile.read())

    cnt=1
    random_selection= random.sample(range(startingPoint, len(dataset_json["results"]["bindings"])), genRnd)
    c=0

    lstDatasetCollection=[]

    for record in dataset_json["results"]["bindings"]:
        if cnt in random_selection or not(rnd):
            c=c+1
            filename=record["fileName"]["value"]
            landingPage=record["dobj"]["value"]
            timeEnd=record["timeEnd"]["value"]
            timeStart=record["timeStart"]["value"]
            submTime=record["submTime"]["value"]
            size =record["size"]["value"]
            spec=record["spec"]["value"]
            #downloadableLink= "https://data.icos-cp.eu/objects/"+os.path.basename(urlparse(landingPage).path)

            lstDatasetCollection.append(landingPage)
        cnt=cnt+1
    return lstDatasetCollection

#----------------------------------------------------------------------------------------
#{"EDMEDRecord": {"type": "uri", "value": "https://edmed.seadatanet.org/report/6325/"},
# "Title": {"type": "literal", "xml:lang": "en", "value": "Water Framework Directive (WFD) Greece (2012-2015)"}}

import random

def processDatasetRecords__SeaDataNet_EDMED(rnd,genRnd):
    indexFile= open(MetaDataRecordPath+SeaDataNet__MetadataRecordsFileName,"r")
    dataset_json = json.loads(indexFile.read())

    cnt=1
    random_selection= random.sample(range(1, len(dataset_json["results"]["bindings"])), genRnd)
    c=0
    lstDatasetCollection=[]

    for record in dataset_json["results"]["bindings"]:
        if cnt in random_selection or not(rnd):
            c=c+1
            landingPage=record["EDMEDRecord"]["value"]
            title=record["Title"]["value"]
            lstDatasetCollection.append(landingPage)
        cnt=cnt+1
    return lstDatasetCollection
#----------------------------------------------------------------------------------------
def processDatasetRecords__SeaDataNet_CDI(rnd,genRnd):
    tree = ET.parse(MetaDataRecordPath+SeaDataNet__MetadataRecordsFileName)
    indexFile = tree.getroot()
    cnt=1
    random_selection= random.sample(range(1, len(indexFile)), genRnd)
    c=0
    lstDatasetCollection=[]
    for record in indexFile:
        if cnt in random_selection or not(rnd):
            c=c+1
            url=record.text
            pos=url.rfind("/xml")
            if(pos and pos+4==len(url)):
                url=url.replace("/xml","/json")
            lstDatasetCollection.append(url)
        cnt=cnt+1
    return lstDatasetCollection

#----------------------------------------------------------------------------------------
def getRI(dataset_JSON):
    RI_content = open(RI_root,"r")
    RI_json = json.loads(RI_content.read())
    dataset_content=extractTextualContent(dataset_JSON)
    for RI in RI_json:
        for RI_keys in RI_json[RI]:
            for ds in dataset_content:
                if RI_keys in ds:
                    return  RI
#----------------------------------------------------------------------------------------
def getDomain(RI_seed):
    domain_content = open(domain_root,"r")
    domain_json = json.loads(domain_content.read())
    for RI in domain_json:
        if RI == RI_seed:
            return domain_json[RI]

#----------------------------------------------------------------------------------------
def extractTextualContent(y):
    out = {}
    lstvalues=[]
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
            text=x
            if type(text)==list or type(text)==dict:
                text=" ".join(str(x) for x in text)
            if type(text)==str and len(text)>1:
                text=re.sub(r'http\S+', '', text)
                if type(text)==str and len(text)>1:
                    lstvalues.append(text)
    flatten(y)
    return lstvalues
#----------------------------------------------------------------------------------------
def clean(doc):
    integer_free = ''.join([i for i in doc if not i.isdigit()])
    stop_free = " ".join([i for i in integer_free.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split() if len(word)>2 and EnglishTerm.check(word))
    return normalized
#----------------------------------------------------------------------------------------
def topicMining(dataset_json):
    lsttopic=[]
    Jsontext=""
    if(dataset_json!=""):
        Jsontext=getContextualText (dataset_json)

        if not len(Jsontext):
            Jsontext=extractTextualContent(dataset_json)
        doc_clean = [clean(doc).split() for doc in Jsontext]
        dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        if len(doc_term_matrix)>0:
            ldamodel = gensim.models.LdaMulticore(corpus=doc_term_matrix,id2word=dictionary,num_topics=3,passes=10)
            topics=ldamodel.show_topics(log=True, formatted=True)
            topTopics= sum([re.findall('"([^"]*)"',listToString(t[1])) for t in topics],[])
            for topic in topTopics:
                lsttopic.append(topic) if topic not in lsttopic else lsttopic
    return lsttopic
#----------------------------------------------------------------------------------------
def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1
#----------------------------------------------------------------------------------------
def getSimilarEssentialVariables(essentialVariables, topics):
    lstEssentialVariables=[]
    lsttopics= [*getSynonyms(topics), *topics]
    for variable in essentialVariables:
        for topic in lsttopics:
            w1=spacy_nlp(topic.lower())
            w2=spacy_nlp(variable.lower())
            similarity=w1.similarity(w2)
            if similarity >= acceptedSimilarityThreshold:
                lstEssentialVariables.append(variable) if variable not in lstEssentialVariables else lstEssentialVariables
    return lstEssentialVariables
#----------------------------------------------------------------------------------------
def getDomainEssentialVariables(domain):
    essentialVariabels_content = open(essentialVariabels_root,"r")
    essentialVariabels_json = json.loads(essentialVariabels_content.read())
    for domainVar in essentialVariabels_json:
        if domain==domainVar:
            return essentialVariabels_json[domain]
#----------------------------------------------------------------------------------------
def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
#----------------------------------------------------------------------------------------
def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v
#----------------------------------------------------------------------------------------
def remove_none(obj):
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return type(obj)((remove_none(k), remove_none(v)) for k, v in obj.items() if k is not None and v is not None)
    else:
        return obj
#----------------------------------------------------------------------------------------
def is_nested_list(l):
    try:
        next(x for x in l if isinstance(x,list))
    except StopIteration:
        return False
    return True
#----------------------------------------------------------------------------------------
def flatten_list(t):
    if(is_nested_list(t)):
        return [str(item) for sublist in t for item in sublist if not type(sublist)==str]
    return t
#----------------------------------------------------------------------------------------
def refineResults(TextArray,datatype,proprtyName):
    datatype=datatype.lower()
    refinedResults=[]
    if len(TextArray):
        if type(TextArray)==str:
            TextArray=[TextArray]

        if type(TextArray)==dict:
            TextArray=list(NestedDictValues(TextArray))

        if type(TextArray)==list:
            TextArray=flatten_list(TextArray)
            values=[]
            for text in TextArray:
                if type(text)==dict:
                    text= list(NestedDictValues(text))
                    values.append(text)
                elif type(text)==list:
                    values=values+text
                else:
                    values.append(text)
            if type (values) == list and len(values):
                TextArray=flatten_list(values)
        if type(TextArray)==type(None):
            TextArray=["\""+str(TextArray)+"\""]
        for text in TextArray:
            doc = spacy_nlp(str(text).strip())
            #..................................................................................
            if ("url" in datatype and type(text)==str):
                urls = re.findall("(?P<url>https?://[^\s]+)", text)
                if len(urls):
                    refinedResults.append(urls) if urls not in refinedResults else refinedResults
            #..................................................................................
            if ("person" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if (len(ent.text)>0) and ent.label_=="PERSON":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("organization" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="ORG":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("place" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="GPE" or ent.label_=="LOC":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("date" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="DATE":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("product" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="PRODUCT":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("integer" in datatype) or ("number" in datatype ):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="CARDINAL":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("money" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="MONEY":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("workofart" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="WORK_OF_ART":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if ("language" in datatype):
                if doc.ents:
                    for ent in doc.ents:
                        if(len(ent.text)>0) and ent.label_=="LANGUAGE" or ent.label_=="GPE":
                            refinedResults.append(ent.text) if ent.text not in refinedResults else refinedResults
            #..................................................................................
            if proprtyName.lower() not in str(text).lower() and ("text" in datatype or "definedterm" in datatype):
                refinedResults.append(text) if text not in refinedResults else refinedResults
            #..................................................................................
    return refinedResults
#----------------------------------------------------------------------------------------
foundResults=[]
def deep_search(needles, haystack):
    found = {}
    if type(needles) != type([]):
        needles = [needles]

    if type(haystack) == type(dict()):
        for needle in needles:
            if needle in haystack.keys():
                found[needle] = haystack[needle]
            elif len(haystack.keys()) > 0:
                #--------------------- Fuzzy calculation
                for key in haystack.keys():
                    if fuzz.ratio(needle.lower(),key.lower()) > 75:
                        found[needle] = haystack[key]
                        break
                #--------------------- ^
                for key in haystack.keys():
                    result = deep_search(needle, haystack[key])
                    if result:
                        for k, v in result.items():
                            found[k] = v
                            foundResults.append(v) if v not in foundResults else foundResults
    elif type(haystack) == type([]):
        for node in haystack:
            result = deep_search(needles, node)
            if result:
                for k, v in result.items():
                    found[k] = v
                    foundResults.append(v) if v not in foundResults else foundResults
    return found
#----------------------------------------------------------------------------------------
def searchField(field,datatype,json):
    foundResults.clear()
    deep_search(field,json)
    refinedResults=refineResults(foundResults,datatype,field)
    return refinedResults
#----------------------------------------------------------------------------------------
def getTopicsByDomainVocabulareis(topics,domain):
    Vocabs=[]
    domainVocbularies_content = open(domainVocbularies_root,"r")
    domainVocbularies_object = json.loads(domainVocbularies_content.read())
    for vocab in domainVocbularies_object[domain]:
        for topic in topics:
            w1=spacy_nlp(topic.lower())
            w2=spacy_nlp(vocab.lower())
            similarity=w1.similarity(w2)
            if similarity > acceptedSimilarityThreshold:
                Vocabs.append(vocab) if vocab not in Vocabs else Vocabs
    return Vocabs
#----------------------------------------------------------------------------------------
def datasetProcessing_ICOS(datasetURL):
    metadataStar_content = open(metadataStar_root,"r")
    metadataStar_object = json.loads(metadataStar_content.read())
    unique_filename = str(uuid.uuid4())
    indexfname = os.path.join(indexFiles_root,"_ICOS_"+unique_filename)
    indexFile= open(indexfname+".json","w+")

    logfile = os.path.join(indexFiles_root,"logfile.csv")
    CSVvalue=""
    if not os.path.exists(logfile):
        logfile= open(logfile,"a+")
        for metadata_property in metadataStar_object:
            CSVvalue=CSVvalue+metadata_property+","
        logfile.write(CSVvalue)
    else:
        logfile= open(logfile,"a+")

    CSVvalue="\n"

    indexFile.write("{\n")

    scripts=Crawler().getHTMLContent(datasetURL,"script")
    for script in scripts:
        if '<script type="application/ld+json">' in str(script):
            script=str(script)
            start = script.find('<script type="application/ld+json">') + len('<script type="application/ld+json">')
            end = script.find("</script>")
            script = script[start:end]
            JSON=json.loads(script)
            RI=""
            domains=""
            topics=[]
            cnt=0
            for metadata_property in metadataStar_object:
                cnt=cnt+1

                if metadata_property=="ResearchInfrastructure":
                    result= getRI(JSON)
                elif metadata_property=="theme":
                    if not len(RI):
                        #RI= getRI(JSON)
                        RI="ICOS"
                    if not len(domains):
                        domains = getDomain(RI)
                    if not len(topics):
                        topics=topicMining(JSON)
                    result=getTopicsByDomainVocabulareis(topics,domains[0])
                elif metadata_property=="potentialTopics":
                    if not len(topics):
                        topics=topicMining(JSON)
                    result=topics
                elif metadata_property=="EssentialVariables":
                    if not len(RI):
                        RI= getRI(JSON)
                    if not len(domains):
                        domains = getDomain(RI)
                    if not len(topics):
                        topics=topicMining(JSON)
                    essentialVariables=getDomainEssentialVariables(domains[0])
                    result=getSimilarEssentialVariables(essentialVariables,topics)
                elif metadata_property=="url":
                    result=datasetURL
                else:
                    result=deep_search([metadata_property],JSON)
                    if not len(result):
                        searchFields=[]
                        for i in range (3, len(metadataStar_object[metadata_property])):
                            result=deep_search([metadataStar_object[metadata_property][i]],JSON)
                            if len(result): searchFields.append(result)
                        result=searchFields
                propertyDatatype=metadataStar_object[metadata_property][0]
                result=refineResults(result,propertyDatatype,metadata_property)

                #if metadata_property=="language" and (result=="" or result==[]):
                 #   result= LangaugePrediction(extractTextualContent(JSON))


                if(cnt==len(metadataStar_object)):
                    extrachar="\n"
                else:
                    extrachar=",\n"

                flattenValue=str(flatten_list(result))
                indexFile.write("\""+str(metadata_property)+"\" :"+flattenValue.replace("'","\"")+extrachar)
                CSVvalue=CSVvalue+flattenValue.replace(",","-").replace("[","").replace("]","").replace("'","").replace("\"","")+","
    logfile.write(CSVvalue)
    indexFile.write("}")
    indexFile.close()
    logfile.close()
#----------------------------------------------------------------------------------------
def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, '', raw_html)
    return  (''.join(x for x in cleantext if x in string.printable)).replace("'","").replace("\"","").strip()
#----------------------------------------------------------------------------------------
lstCoveredFeaturesSeaDataNet=[]
def getValueHTML_SeaDataNet(searchTerm, datasetContents):
    for datasetContent in datasetContents:
        datasetContent=str(datasetContent)
        if searchTerm in datasetContent and searchTerm not in lstCoveredFeaturesSeaDataNet:
            lstCoveredFeaturesSeaDataNet.append(searchTerm)
            return cleanhtml(datasetContent)[len(searchTerm):]
#----------------------------------------------------------------------------------------
def datasetProcessing_SeaDataNet_EDMED(datasetURL):
    metadataStar_content = open(metadataStar_root,"r")
    metadataStar_object = json.loads(metadataStar_content.read())
    unique_filename = str(uuid.uuid4())
    indexfname = os.path.join(indexFiles_root,"_SeaDataNet_EDMED_"+unique_filename)
    indexFile= open(indexfname+".json","w+")
    logfile = os.path.join(indexFiles_root,"logfile.csv")
    CSVvalue=""
    if not os.path.exists(logfile):
        logfile= open(logfile,"a+")
        for metadata_property in metadataStar_object:
            CSVvalue=CSVvalue+metadata_property+","
        logfile.write(CSVvalue)
    else:
        logfile= open(logfile,"a+")

    indexFile.write("{\n")
    datasetContents=Crawler().getHTMLContent(datasetURL,"tr")
    lstCoveredFeaturesSeaDataNet.clear()
    mapping={}
    cnt=0
    TextualContents=""

    mapping["url"]=str(datasetURL)
    mapping["ResearchInfrastructure"]="SeaDataNet"
    RI="SeaDataNet"

    value=(getValueHTML_SeaDataNet("Data set name", datasetContents))
    mapping["name"]=str(value)

    value=(getValueHTML_SeaDataNet("Data holding centre", datasetContents))
    mapping["copyrightHolder"]=str(value)
    mapping["contributor"]=str(value)

    value=(getValueHTML_SeaDataNet("Country", datasetContents))
    mapping["locationCreated"]=str(value)
    mapping["contentLocation"]=str(value)

    value=(getValueHTML_SeaDataNet("Time period", datasetContents))
    mapping["contentReferenceTime"]=str(value)
    mapping["datePublished"]=str(value)
    mapping["dateCreated"]=str(value)

    value=(getValueHTML_SeaDataNet("Geographical area", datasetContents))
    mapping["spatialCoverage"]=str(value)

    value=(getValueHTML_SeaDataNet("Parameters", datasetContents))
    mapping["keywords"]=str(value)

    value=(getValueHTML_SeaDataNet("Instruments", datasetContents))
    mapping["measurementTechnique"]=str(value)

    value=(getValueHTML_SeaDataNet("Summary", datasetContents))
    mapping["description"]=str(value)
    mapping["abstract"]=str(value)
    TextualContents=value

    #mapping["language"]=[LangaugePrediction(TextualContents)]

    value=(getValueHTML_SeaDataNet("Originators", datasetContents))
    mapping["creator"]=str(value)

    value=(getValueHTML_SeaDataNet("Data web site", datasetContents))
    mapping["distributionInfo"]=str(value)

    value=(getValueHTML_SeaDataNet("Organisation", datasetContents))
    mapping["publisher"]=str(value)

    value=(getValueHTML_SeaDataNet("Contact", datasetContents))
    mapping["author"]=str(value)

    value=(getValueHTML_SeaDataNet("Address", datasetContents))
    mapping["contact"]=str(value)

    value=(getValueHTML_SeaDataNet("Collating centre", datasetContents))
    mapping["producer"]=str(value)
    mapping["provider"]=str(value)

    value=(getValueHTML_SeaDataNet("Local identifier", datasetContents))
    mapping["identifier"]=str(value)

    value=(getValueHTML_SeaDataNet("Last revised", datasetContents))
    mapping["modificationDate"]=str(value)

    domains = getDomain(RI)
    value=topicMining(TextualContents)
    mapping["potentialTopics"]=value

    essentialVariables=getDomainEssentialVariables(domains[0])
    value=getSimilarEssentialVariables(essentialVariables,value)
    mapping["EssentialVariables"]=value

    CSVvalue="\n"

    for metadata_property in metadataStar_object:
        cnt=cnt+1
        if(cnt==len(metadataStar_object)):
            extrachar="\n"
        else:
            extrachar=",\n"

        if metadata_property in mapping:
            value=mapping[metadata_property]
            if type( mapping[metadata_property])!=list:
                value=[value]
            indexFile.write("\""+str(metadata_property)+"\" :"+str(value).replace("'","\"")+extrachar)
            CSVvalue=CSVvalue+str(value).replace(",","-").replace("[","").replace("]","").replace("'","").replace("\"","")+","
        else:
            indexFile.write("\""+str(metadata_property)+"\" :"+ str([])+extrachar)
            CSVvalue=CSVvalue+","

#    value=(getValueHTML_SeaDataNet("Availability", datasetContents))
#    value=(getValueHTML_SeaDataNet("Ongoing", datasetContents))
#    value=(getValueHTML_SeaDataNet("Global identifier", datasetContents))
    logfile.write(CSVvalue)
    indexFile.write("}")
    indexFile.close()
    logfile.close()
#----------------------------------------------------------------------------------------
def NestedDictValues(d):
    if type(d)==dict:
        for v in d.values():
            if isinstance(v, dict):
                yield from NestedDictValues(v)
            else:
                yield v
#----------------------------------------------------------------------------------------
def MergeList(contextualText):
    lstText=[]
    for entity in contextualText:
        if type(entity)==list:
            for item in entity:
                lstText.append(item.strip())
        else:
            lstText.append(entity.strip())
    return lstText
#----------------------------------------------------------------------------------------
def getContextualText(JSON):
    contextualText=""
    contextualText=deep_search(["Data set name", "Discipline","Parameter groups","Discovery parameter","GEMET-INSPIRE themes"],JSON)
    if not len(contextualText):
        contextualText=deep_search(["Abstract"],JSON)
    contextualText=list(NestedDictValues(contextualText))
    return MergeList(contextualText)
#----------------------------------------------------------------------------------------
def datasetProcessing_SeaDataNet_CDI(datasetURL):
    metadataStar_content = open(metadataStar_root,"r")
    metadataStar_object = json.loads(metadataStar_content.read())
    with urllib.request.urlopen(datasetURL) as f:
        data = f.read().decode('utf-8')
    JSON=json.loads(data)

    unique_filename = str(uuid.uuid4())
    indexfname = os.path.join(indexFiles_root,"SeaDataNet_CDI_"+unique_filename)
    indexFile= open(indexfname+".json","w+")

    logfile = os.path.join(indexFiles_root,"logfile.csv")
    CSVvalue=""
    if not os.path.exists(logfile):
        logfile= open(logfile,"a+")
        for metadata_property in metadataStar_object:
            CSVvalue=CSVvalue+metadata_property+","
        logfile.write(CSVvalue)
    else:
        logfile= open(logfile,"a+")

    CSVvalue="\n"

    indexFile.write("{\n")

    RI=""
    domains=""
    topics=[]
    cnt=0
    for metadata_property in metadataStar_object:
        cnt=cnt+1

        if metadata_property=="ResearchInfrastructure":
            result= getRI(JSON)
        elif metadata_property=="theme":
            if not len(RI):
                #RI= getRI(JSON)
                RI="SeaDataNet"
            if not len(domains):
                domains = getDomain(RI)
            if not len(topics):
                topics=topicMining(JSON)
            result=getTopicsByDomainVocabulareis(topics,domains[0])
        elif metadata_property=="language":
            result="English"
        elif metadata_property=="potentialTopics":
            if not len(topics):
                topics=topicMining(JSON)
            result=topics
        elif metadata_property=="EssentialVariables":
            if not len(RI):
                RI= getRI(JSON)
            if not len(domains):
                domains = getDomain(RI)
            if not len(topics):
                topics=topicMining(JSON)
            essentialVariables=getDomainEssentialVariables(domains[0])
            result=getSimilarEssentialVariables(essentialVariables,topics)
        elif metadata_property=="url":
            result=datasetURL
        else:
            result=deep_search([metadata_property],JSON)
            if not len(result):
                searchFields=[]
                for i in range (3, len(metadataStar_object[metadata_property])):
                    result=deep_search([metadataStar_object[metadata_property][i]],JSON)
                    if len(result): searchFields.append(result)
                result=searchFields
        propertyDatatype=metadataStar_object[metadata_property][0]
        result=refineResults(result,propertyDatatype,metadata_property)

        #if metadata_property=="language" and (result=="" or result==[]):
        #   result= LangaugePrediction(extractTextualContent(JSON))


        if(cnt==len(metadataStar_object)):
            extrachar="\n"
        else:
            extrachar=",\n"

        flattenValue=(str(MergeList(flatten_list(result)))
                          .replace("></a","").replace(",","-")
                          .replace("[","").replace("]","").replace("{","")
                          .replace("'","").replace("\"","").replace("}","")
                          .replace("\"\"","").replace(">\\","")
                          .replace("' ","'").replace(" '","'"))
        flattenValue= str([x.strip() for x in flattenValue.split('-')])

        indexFile.write("\""+str(metadata_property)+"\" :"+flattenValue.replace("'","\"")+extrachar)
        CSVvalue=CSVvalue+flattenValue.replace(",","-").replace("[","").replace("]","").replace("'","").replace("\"","").replace("\"\"","")+","
    logfile.write(CSVvalue)
    indexFile.write("}")
    indexFile.close()
    logfile.close()
#----------------------------------------------------------------------------------------
class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, int):
            try:
                return str(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o
#----------------------------------------------------------------------------------------
def datasetProcessing_SeaDataNet_CDI_XML(datasetURL):
    metadataStar_content = open(metadataStar_root,"r")
    metadataStar_object = json.loads(metadataStar_content.read())

    with urllib.request.urlopen(datasetURL) as f:
        data = f.read().decode('utf-8')
    data=data.replace("gmd:","").replace("gco:","").replace("sdn:","").replace("gml:","").replace("gts:","").replace("xlink:","").replace("\"{","")
    xml = fromstring(data.encode())
    JSON=json.loads(json.dumps(xmljson.badgerfish.data(xml)),cls=Decoder)
#----------------------------------------------------------------------------------------
#--------------------
#getDataSetRecords__ICOS()
#getDatasetRecords__SeaDataNet_EDMED()
#getDatasetRecords__SeaDataNet_CDI()
#--------------------
lstDataset= processDatasetRecords__SeaDataNet_CDI(True,1)
for datasetURL in lstDataset:
    datasetProcessing_SeaDataNet_CDI(datasetURL)


#--------------------

#lstDataset= (processDatasetRecords__ICOS(True,2000,250000))
#for datasetURL in lstDataset:
#    datasetProcessing_ICOS(datasetURL)

#datasetProcessing_ICOS("https://meta.icos-cp.eu/objects/Msxml8TlWbHvmQmDD6EdVgPc")

#datasetProcessing_ICOS("https://meta.icos-cp.eu/objects/7c3iQ3A8SAeupVvMi8wFPWEN")


#lstDataset= (processDatasetRecords__SeaDataNet(True,1000))
#for datasetURL in lstDataset:
#   datasetProcessing_SeaDataNet(datasetURL)

#datasetProcessing_SeaDataNet("https://edmed.seadatanet.org/report/249/")
#--------------------




