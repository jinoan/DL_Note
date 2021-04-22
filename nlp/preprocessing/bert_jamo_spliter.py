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
            return c
        jamo = decompose(c)
        # 'a' or 모음 or 자음
        if jamo is None:
            return ' '
        return ''.join(jamo).strip()
    
    s = ''.join(re.compile('[0-9|a-z|A-Z|ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?!|\s]+').findall(s))
    s = findrepeat(s)
    s = ''.join(process(c) for c in s)
    return remove_doublespace(s).strip()

def decode(s):
    def process(w):
        chars = []
        temp = ''
        for c in w:
            if re.compile('[ㄱ-ㅎ]').match(c):
                if len(temp) == 0:
                    temp = c
                elif len(temp) == 1:
                    chars.append(' '+temp+' ')
                    temp = c
                elif len(temp) == 2:
                    temp += c
                else:
                    chars.append(temp)
                    temp = c
            elif re.compile('[ㅏ-ㅣ]').match(c):
                if len(temp) == 0:
                    chars.append(' '+c+' ')
                elif len(temp) == 1:
                    temp += c
                elif len(temp) == 2:
                    chars.append(temp+' ')
                    chars.append(' '+c+' ')
                    temp = ''
                else:
                    chars.append(temp[:2]+' ')
                    temp = temp[2]+c
            else:
                if len(temp) > 0:
                    if len(temp) == 1:
                        chars.append(' '+temp+' ')
                    elif len(temp) == 2:
                        chars.append(temp+' ')
                    else:
                        chars.append(temp)
                    temp = ''
                chars.append(c)
        else:
            if len(temp) == 1:
                chars.append(' '+temp+' ')
            elif len(temp) == 2:
                chars.append(temp+' ')
            elif len(temp) == 3:
                chars.append(temp)
        recovered = []
        for char in chars:
            if len(char) == 3:
                recovered.append(compose(*char))
            else:
                recovered.append(char)
        recovered = ''.join(recovered)
        return recovered
    return ' '.join(process(t) for t in s.split())
