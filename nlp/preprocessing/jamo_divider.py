# -*- coding: utf-8 -*-

import re
from soynlp.hangle import decompose, compose

def remove_doublespace(s):
    doublespace_pattern = re.compile('\s+')
    return doublespace_pattern.sub(' ', s).strip()

def findrepeat(text):
    for t in set([c for c in text]):
        for s, e in reversed([(m.start(), m.end()) for m in re.compile('['+t+']{3,}').finditer(text)]):
             text = text[:s] + t*3 + text[e:]
    return text

def encode(s):
    def process(c):
        if re.compile('[0-9|a-z|A-Z|.?!]+').match(c):
            return '-' + c + '-'
        jamo = decompose(c)
        # 'a' or 모음 or 자음
        if (jamo is None):
            return ' '
        cho, jung, jong = (jamo)
        if jong == ' ':
            jong = '-'
            if jung == ' ':
                return '-' + cho + '-'
            else:
                if cho == ' ':
                    cho = '-'
        return cho + jung + jong
    
    s = ''.join(re.compile('[0-9|a-z|A-Z|ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?!|\s]+').findall(s))
    s = findrepeat(s)
    s = ''.join(process(c) for c in s)
    return remove_doublespace(s).strip()

def decode(s):
    def process(t):
        assert len(t) % 3 == 0
        t_ = t.replace('-', ' ')
        chars = [tuple(t_[3*i:3*(i+1)]) for i in range(len(t_)//3)] 
        recovered = list()
        for char in chars:
            try:
                recovered.append(compose(*char))
            except ValueError:
                recovered.append(''.join(char).replace('-', '').strip())
        recovered = ''.join(recovered)
        return recovered

    return ' '.join(process(t) for t in s.split())
