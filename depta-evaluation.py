# -*- coding: utf-8 -*-
from lxml import html, etree
import os
from io import open

from pydepta.depta import Depta

def elm2text(e):
    textElm = [x for x in e.xpath('descendant-or-self::*[@data-index]') if x.text is not None] # only consider elements that are touched
    return ' '.join([x.text.strip() for x in textElm]).strip()

def extract_records(inputFolder, outputFolder):
    d = Depta()
    for file in os.listdir(inputFolder):
        if file.split('.')[-1] != 'html':
            continue
        input = open(os.path.join(inputFolder, file))
        regions = d.extract(input.read())
        input.close()
        output = open(os.path.join(outputFolder, file.split('.')[0] + '.txt'), 'wb')
        for region in regions:
            cnt = 0
            for record in region.records:
                recordText = []
                for element in record.elements:
                    elementTexts = []
                    for s in element.itertext():
                        t = ' '.join(s.replace('|', '').split())
                        if len(t) > 0:
                            elementTexts.append(t)
                    if len(elementTexts) > 0:
                        recordText.extend(elementTexts)
                if len(recordText) > 0:
                    cnt += 1
                    line = '|'.join(recordText) + '\n'
                    output.write(line.encode('utf-8'))
            if cnt > 0:
                output.write('\n'.encode('utf-8'))
        output.close()
        print('Finished ' + file)

# def score(dom, annotationXpath, recordRegions):
#     annotatedElement = dom.xpath(annotationXpath)
#     index2text = {}
#     annotatedTexts = set()
#     for e in annotatedElement:
#         t = elm2text(e)
#         index2text[int(e.attrib['data-index'])] = t
#         annotatedTexts.add(t)

#     for recordIndexes in recordGroups:
#         for i in recordIndexes:
#             if i not in index2text:
#                 t = elm2text(sTree[i].elm)
#                 index2text[i] = t

#     # only evaluate records that found with ground truth
#     detectedIndexes = set()
#     for recordIndexes in recordGroups:
#         groupTexts = set([elm2text(sTree[i].elm) for i in recordIndexes])
#         if len(groupTexts.intersection(annotatedTexts)) > 0:
#             detectedIndexes.update(set(recordIndexes))
    
#     hit = miss = mistake = 0
#     detectedTexts = set()
#     for i in detectedIndexes:
#         t = index2text[i]
#         detectedTexts.add(t)
#         if len(t) == 0:
#             continue
#         if t in annotatedTexts:
#             sTree[i].elm.attrib["style"] = "border: thick solid darkgreen;"
#             hit += 1
#         else:
#             sTree[i].elm.attrib["style"] = "border: thick solid yellow;"
#             mistake += 1
#     for e in annotatedElement:
#         t = index2text[int(e.attrib['data-index'])]
#         if len(t) == 0:
#             continue
#         if t not in detectedTexts:
#             e.attrib["style"] = "border: thick solid red;"
#             miss += 1
#     return hit, miss, mistake
for dataset in ['google', 'amazon']:
    commentInputFolder = os.path.join('data', dataset)
    commentOutputFolder = os.path.join('data', '{}-result-depta'.format(dataset))
    extract_records(commentInputFolder, commentOutputFolder)