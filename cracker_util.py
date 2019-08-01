from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import re
import string

def getDict(merge = True):
    if not merge: 
        from data_cracker import GTUtility
        gt_util = GTUtility('data/KSIGNBOARD/')
    if merge:
        from data_KSign import GTUtility
        gt_util_high = GTUtility('data/K-SIGN/HighQuality/', polygon=False, quality='high')
        
        from data_KSign import GTUtility
        gt_util_ai_cr = GTUtility('data/K-SIGN/Annotation/', polygon=False, quality='cr')
        
        from data_cracker import GTUtility
        gt_util_cr = GTUtility('data/KSIGNBOARD/')
        
        gt_util = gt_util_cr.merge(gt_util_high)
        gt_util = gt_util.merge(gt_util_ai_cr)

    text = gt_util.text
    text = list(chain(*text))
    vect = CountVectorizer(analyzer='char').fit(text)
    charset = list(vect.vocabulary_.keys())
    pattern = '[^가-힣]' #한글이 아닌 문자는 공백으로 바꿔준다
    charset_dict = [re.sub(pattern, "", char) for char in charset]
    charset_dict2 = [x for x in charset_dict if x!= '']
    dict_str = "".join(charset_dict2)
    dict_str = dict_str + string.ascii_lowercase + string.ascii_uppercase
    dict_str = dict_str + string.digits + ' +-*.,:!?%&$~/()[]<>"\'@#_'      
    
    return dict_str

cracker_dict = getDict()

def decode(chars):
    blank_char = '_'
    new = ''
    last = blank_char
    for c in chars:
        if (last == blank_char or last != c) and c != blank_char:
            new += c
        last = c
    return new
