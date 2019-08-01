import json
import os
from data_KSign import GTUtility
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
import re
import string

def getDict239():
    gt_util = GTUtility('data/K-Sign/HighQuality/', quality='high')
    text = gt_util.text
    text = list(chain(*text))
    vect = CountVectorizer(analyzer='char').fit(text)
    charset = list(vect.vocabulary_.keys())
    
    pattern = '[^가-힣]' #한글이 아닌 문자는 공백으로 바꿔준다
    charset_dict = [re.sub(pattern, "", char) for char in charset]
    ksign_dict = [x for x in charset_dict if x!= '']
    ksign_dict = "".join(ksign_dict)
    ksign_dict = ksign_dict + string.digits + ' _'
    
    return ksign_dict

ksign_dict = getDict239()