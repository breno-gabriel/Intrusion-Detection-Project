"""Generates data for train/test algorithms"""
from datetime import datetime
from io import StringIO  
# from urllib import urlopen
from zipfile import ZipFile
import pickle
import os
import random
import tldextract
import csv 
import random

from dga_classifier.dga_generators import banjori, corebot, cryptolocker, \
    dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda

# Location of Alexa 1M
# ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

# Our ourput file containg all the training data
DATA_FILE = 'traindata.pkl'

# def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):
#     """Grabs Alexa 1M"""
#     url = urlopen(address)
#     zipfile = ZipFile(StringIO(url.read()))
#     return [tldextract.extract(x.split(',')[1]).domain for x in \
#             zipfile.read(filename).split()[:num]]

import csv

def get_majestic_domain(num, filename='majestic_million.csv'):
    with open(filename, 'r', encoding='utf-8') as arquivo:
        arquivo_csv = csv.reader(arquivo, delimiter=",")
        
        dominios = []

        for linha in arquivo_csv:
            dominio = linha[2]
            dominios.append(dominio)  

            if len(dominios) == num:
                break

    return dominios

def get_data(num_per_dga=15343):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []

    # We use some arbitrary seeds to create domains with banjori
    banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                     'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                     'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                     'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                     'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                     'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                     'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                     'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                     'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                     'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                     'atlanta', 'portland', 'seattle', 'washingtondc']

    segs_size = max(1, round(num_per_dga/len(banjori_seeds)))
    for banjori_seed in banjori_seeds:
        res =  banjori.generate_domains(segs_size, banjori_seed)
        domains += res
        labels += ['banjori']*len(res)

    res = corebot.generate_domains(num_per_dga)
    domains += res
    labels += ['corebot']*len(res)

    # Create different length domains using cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = max(1, round(num_per_dga/len(crypto_lengths)))
    for crypto_length in crypto_lengths:
        res = cryptolocker.generate_domains(segs_size,
                                                 seed_num=random.randint(1, 1000000),
                                                 length=crypto_length)
        domains += res 
        labels += ['cryptolocker']*len(res)

    res =  dircrypt.generate_domains(num_per_dga)
    domains += res
    labels += ['dircrypt']*len(res)

    # generate kraken and divide between configs
    kraken_to_gen = max(1, round(num_per_dga/2))
    res = kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'a', 3)
    domains += res
    labels += ['kraken']*len(res)
    res = kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'b', 3)
    domains += res
    labels += ['kraken']*len(res)

    # generate locky and divide between configs
    locky_gen = max(1, round(num_per_dga/11))
    for i in range(1, 12):
        res = lockyv2.generate_domains(locky_gen, config=i)
        domains += res
        labels += ['locky']*len(res)

    # Generate pyskpa domains
    res = pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    domains += res
    labels += ['pykspa']*len(res)

    # Generate qakbot
    res = qakbot.generate_domains(num_per_dga, tlds=[])
    domains += res
    labels += ['qakbot']*len(res)

    # ramdo divided over different lengths
    ramdo_lengths = range(8, 32)
    segs_size = max(1, round(num_per_dga/len(ramdo_lengths)))
    for rammdo_length in ramdo_lengths:
        res =  ramdo.generate_domains(segs_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=rammdo_length)
        domains += res
        labels += ['ramdo']*len(res)

    # ramnit
    res = ramnit.generate_domains(num_per_dga, 0x123abc12)
    domains += res
    labels += ['ramnit']*len(res)

    # simda
    simda_lengths = range(8, 32)
    segs_size = max(1, round(num_per_dga/len(simda_lengths)))
    for simda_length in range(len(simda_lengths)):
        res = simda.generate_domains(segs_size,
                                          length=simda_length,
                                          tld=None,
                                          base=random.randint(2, 2**32))
        domains += res 
        labels += ['simda']*len(res)

    res = get_majestic_domain(len(domains))
    domains += res
    labels += ['benign']*len(res)

    # Embaralhe aleatoriamente os domínios e mantenha as correspondências com os rótulos
    combined_data = list(zip(domains, labels))
    random.shuffle(combined_data)
    domains, labels = zip(*combined_data)

    # print(len(domains))
    # print(len(labels))

    return domains, labels


