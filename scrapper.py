import time
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
from scrapy import Selector
import threading
import multiprocess
import csv


path = 'C:\\Users\\Diogo Gonçalves\\Documents\\Ironhack\\final-project\\Data\\'
df_basics = pickle.load(open(path+"title.basics.sav","rb"))
df = df_basics[df_basics['titleType']=='movie']
df.head()