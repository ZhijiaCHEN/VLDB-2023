from collections import defaultdict
import json
from lxml import html, etree
from lxml.html.clean import Cleaner
import pandas as pd
from univeral_tree import StructNode, build_lxml_tree, StructTree
import os, time, random
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import process_time
from zss import simple_distance, Node
from bisect import bisect
from os.path import join
import statistics
ANNOTATION_PATH = os.path.join('data', 'comments', 'annotation-patch')
NAGATIVE_PATH = "D:/Google Drive/Temple/projects/comment entry/data/comment/negative/"
OUTPUT_PATH = os.path.join('data', 'comments', 'cross-validation')

TBDW_INPUT_PATH = os.path.join('D:', os.sep, 'Downloads', 'TBDW-mod')
TBDW_OUTPUT_PATH = os.path.join('D:', os.sep, 'Downloads', 'TBDW-evaluation')

L_RANGE = list(range(1,11))
F_RANGE = list(range(2,11))
font = {'family' : 'Times New Roman',
            'size'   : 12}
matplotlib.rc('font', **font)
def test(onePerWebsite = True, freqThresh = 3, lenThresh = 3, method = StructTree.STRUCT_PATTERN, topWorst = False, inputPath = ANNOTATION_PATH, outputPath = None, annotation="data-record-boundary", skipSite = []):
    if outputPath is not None:
        for file in os.listdir(OUTPUT_PATH):
            os.remove(os.path.join(OUTPUT_PATH, file))
    siteSet = set()
    hitTotal, missTotal, mistakeTotal = 0, 0, 0
    webMissCnt = defaultdict(int)
    webMistakeCnt = defaultdict(int)
    webHitCnt = defaultdict(int)
    webPrecision = {}
    webRecall = {}
    for file in os.listdir(inputPath):
        if file.split('.')[-1] != 'html':
            continue
        if len(file.split('_')) > 1:
            siteID = file.split('_')[1].split('-')[0]
        else:
            siteID = file.split('-')[0]
        if int(siteID) in skipSite:
            continue
        if onePerWebsite:
            if siteID in siteSet:
                continue
            siteSet.add(siteID)
        print(f"Processing {file}")
        hit, miss, mistake = test_one(os.path.join(inputPath, file), freqThresh=freqThresh, lenThresh=lenThresh, method=method, outputPath=outputPath, annotation=annotation)
        if hit < 0: continue
        webMissCnt[siteID] += miss
        webMistakeCnt[siteID] += mistake
        webHitCnt[siteID] += hit
        hitTotal += hit
        missTotal += miss
        mistakeTotal += mistake
        # if hit + miss > 0:
        #     recall = hit / (hit + miss)
        # else:
        #     recall = 0
        # if hit + mistake > 0:
        #     precision = hit / (hit + mistake)
        # else:
        #     precision = 0
            
    if topWorst:
        for wid in webHitCnt:
            hit = webHitCnt[wid]
            miss = webMissCnt[wid]
            mistake = webMistakeCnt[wid]
            if hit + mistake > 0:
                webPrecision[wid] = hit / (hit + mistake)
            else:
                webPrecision[wid] = 0
            if hit + miss > 0:
                webRecall[wid] = hit / (hit + miss)
            else:
                webRecall[wid] = 0
        webMistakeCnt = sorted([(cnt, wid) for wid, cnt in webMistakeCnt.items()], reverse=True)
        webMissCnt = sorted([(cnt, wid) for wid, cnt in webMissCnt.items()], reverse=True)
        # webPrecision = sorted([(precision, wid) for wid, precision in webPrecision.items()])
        # webRecall = sorted([(recall, wid) for wid, recall in webRecall.items()])
        print('Top 10 miss websites: ')
        for cnt, wid in webMissCnt[:10]:
            print(f'>>{wid}: miss = {cnt}, precision = {webPrecision[wid]}, recall = {webRecall[wid]}')
        print('Top 10 mistake websites: ')
        for cnt, wid in webMistakeCnt[:10]:
            print(f'>>{wid}: mistake = {cnt}, precision = {webPrecision[wid]}, recall = {webRecall[wid]}')

    recallTotal = hitTotal/(hitTotal + missTotal)
    precisionTotal = hitTotal/(hitTotal + mistakeTotal)
    print(f'Precision = {precisionTotal}, recall = {recallTotal}.')
    return precisionTotal, recallTotal

def test_one(inputFile, freqThresh = 3, lenThresh = 3, method=StructTree.STRUCT_PATTERN, outputPath = None, annotation = "data-record-boundary"):
    with open(inputFile, encoding='utf-8') as f:
        hit, miss, mistake = 0, 0, 0
        eTree = build_lxml_tree(f)
        sTree = StructTree(eTree, pattern_method=method)
        annotated = eTree.xpath(f"//*[@{annotation}]")
        annotatedIndex = set([int(a.attrib["data-index"]) for a in annotated])
        if len(annotated) == 0:
            print(f'No comment boundary labeled in {inputFile}.')
            return -1, -1, -1
        targetRecordRegion = sTree._lowest_common_ancestor(annotatedIndex)
        recordRegion = sTree.record_boundary(lenThresh, freqThresh, 2, 3)
        # recordIndexes = [recordIdx for kernel2Record in recordRegion.values() for recordIdx in kernel2Record.values()]
        
        # remove elements not within target record region
        # detected = []
        # for i in recordIndexes:
        #     x = sTree[i]
        #     while x:
        #         if x.index == targetRecordRegion:
        #             detected.append(sTree[i].elm)
        #             break
        #         x = x.parent

        detected = [sTree[recordIdx].elm for kernel2Record in recordRegion.values() for recordIdx in kernel2Record.values()]
        id2object = {id(x):x for x in detected}
        id2object.update({id(y): y for y in annotated})
        annotatedID = set([id(x) for x in annotated])
        detectedID = set([id(x) for x in detected])
        for x in detectedID:
            # if len(id2object[x].text_content().replace('\n','').replace('\t', '').replace(' ', '')) == 0:
            #     continue
            e = id2object[x]
            if x in annotatedID:
                e.attrib["style"] = "border: thick solid darkgreen;"
                hit += 1
            else:
                e.attrib["style"] = "border: thick solid yellow;"
                mistake += 1
        for x in annotatedID:
            e = id2object[x]
            if x not in detectedID:
                e.attrib["style"] = "border: thick solid red;"
                miss += 1
        if outputPath is not None:
            with open(os.path.join(outputPath, inputFile.split(os.sep)[-1]), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
        return hit, miss, mistake

def sensitivity_analysis():
    for method in [StructTree.HTP_PATTERN, StructTree.TAG_ATTRIB_PATTERN, StructTree.STRUCT_PATTERN]:
        precisionMesh = pd.DataFrame(index=F_RANGE, columns=L_RANGE)
        recallMesh = pd.DataFrame(index=F_RANGE, columns=L_RANGE)
        for freqThresh in F_RANGE:
            for lenThresh in L_RANGE:
                print(f'Testing {method=}, {lenThresh=}, {freqThresh=}.')
                precision, recall = test(onePerWebsite=True, freqThresh=freqThresh, lenThresh=lenThresh, method=method)
                precisionMesh.loc[freqThresh, lenThresh] = precision
                recallMesh.loc[freqThresh, lenThresh] = recall
        precisionMesh.to_csv(f'sensitivity-{method}-precision.csv', float_format='%.4f')
        recallMesh.to_csv(f'sensitivity-{method}-recall.csv', float_format='%.4f')

def plot_sensitivity():
    precision = pd.read_csv('sensitivity-STRUCTURE_ENCODING-precision.csv', index_col=0).transpose()
    recall = pd.read_csv('sensitivity-STRUCTURE_ENCODING-recall.csv', index_col=0).transpose()
    f1 = 2*precision*recall/(precision + recall)
    # precision -= precision.min().min()
    
    plt.figure(figsize=(6, 3))
    xtickIndex = list(range(0, len(F_RANGE), 2))
    xtickLabel = [F_RANGE[i] for i in xtickIndex]
    ytickIndex = list(range(0, len(L_RANGE), 2))
    ytickLabel = [L_RANGE[i] for i in ytickIndex]
    fig, ax = plt.subplots(1,3)
    # ax[0].pcolormesh(precision)
    im = ax[0].imshow(precision)
    ax[0].set_xlabel('$F_{th}$')
    ax[0].set_ylabel('$L_{th}$')
    ax[0].set_xticks(ticks=xtickIndex, labels=xtickLabel)
    ax[0].set_yticks(ticks=ytickIndex, labels=ytickLabel)
    ax[0].set_title('Precision', x=0.5, y=1.35)
    divider = make_axes_locatable(ax[0])
    cax = divider.new_vertical(size='5%', pad=0.4)
    fig.add_axes(cax)
    
    minVal = precision.min().min()
    maxVal = precision.max().max()
    midVal = (minVal + maxVal)/2
    delta = (maxVal - minVal)/10
    fig.colorbar(im, orientation="horizontal", cax=cax, ticks=[round(minVal+delta, 3), round(midVal, 3), round(maxVal-delta, 3)])

    im = ax[1].imshow(recall)
    ax[1].set_xlabel('$F_{th}$')
    # plt.ylabel('$L_{th}$')
    ax[1].set_xticks(ticks=xtickIndex, labels=xtickLabel)
    # plt.yticks(ticks=ytickIndex, labels=ytickLabel)
    ax[1].set_yticks(ticks=[])
    ax[1].set_title('Recall', x=0.5, y=1.35)
    divider = make_axes_locatable(ax[1])
    cax = divider.new_vertical(size='5%', pad=0.4)
    fig.add_axes(cax)
    
    minVal = recall.min().min()
    maxVal = recall.max().max()
    midVal = (minVal + maxVal)/2
    delta = (maxVal - minVal)/10
    fig.colorbar(im, orientation="horizontal", cax=cax, ticks=[round(minVal+delta, 3), round(midVal, 3), round(maxVal-delta, 3)])

    im = ax[2].imshow(f1)
    # ax[2].pcolormesh(f1)
    # ax[2].colorbar(orientation="horizontal", location="top")
    ax[2].set_xlabel('$F_{th}$')
    # plt.ylabel('$L_{th}$')
    ax[2].set_xticks(ticks=xtickIndex, labels=xtickLabel)
    # plt.yticks(ticks=ytickIndex, labels=ytickLabel)
    ax[2].set_yticks(ticks=[])
    ax[2].set_title('F1', x=0.5, y=1.35)
    divider = make_axes_locatable(ax[2])
    cax = divider.new_vertical(size='5%', pad=0.4)
    fig.add_axes(cax)
    
    minVal = f1.min().min()
    maxVal = f1.max().max()
    midVal = (minVal + maxVal)/2
    delta = (maxVal - minVal)/10
    fig.colorbar(im, orientation="horizontal", cax=cax, ticks=[round(minVal+delta, 3), round(midVal, 3), round(maxVal-delta, 3)])
    plt.tight_layout()
    plt.show()

def size_vs_time(testFun):
    X = []
    Y = []
    for file in os.listdir(ANNOTATION_PATH):
        if file.split('.')[-1] != 'html':
            continue
        with open(os.path.join(ANNOTATION_PATH, file), encoding='utf-8') as f:
            eTree = build_lxml_tree(f)
            X.append(len(eTree.xpath('.//*')))
        startTime = process_time()
        testFun(file)
        endTime = process_time()
        Y.append(endTime - startTime)
    return X, Y

def plot_efficiency():
    data = pd.read_csv('size-vs-time.csv', index_col='file')
    fig, ax = plt.subplots(1,1)
    alpha = 1
    size = 5
    labelFont = {}
    ax.scatter(data['size'], data['date-pattern'], label='MiBAT', alpha=alpha, s=size, marker='v')
    ax.scatter(data['size'], data['depta'], label = 'DEPTA', alpha=alpha, s=size, marker='s')
    ax.scatter(data['size'], data['struct-pattern'], label='$Miria$', alpha=alpha, s=size, marker='o')
    
    
    ax.set_xlim([0, 5000])
    ax.set_ylim([0, 8])
    ax.set_yticks(ticks=list(range(2,9,2)))
    ax.set_xlabel('size of DOM tree')
    ax.set_ylabel('processing time (s)')
    plt.gcf().set_size_inches(4, 2)
    # ax[0].grid()
    # ax[0].legend(loc='upper left')
    # ax[1].scatter(data['size'], data['date-pattern'], label='date', alpha=0.3, s=1)
    # ax[1].set_xlim([0, 5000])
    # ax[1].set_ylim([0, 8])
    # ax[1].set_xticks(ticks=[])
    # ax[1].set_yticks(ticks=list(range(2,9,2)))
    # ax[1].grid()
    # ax[1].legend(loc='upper left')
    # ax[2].scatter(data['size'], data['depta'], label = 'depta', alpha=0.3, s=1)
    # ax[2].set_xlim([0, 5000])
    # ax[2].set_ylim([0, 8])
    # ax[2].set_yticks(ticks=list(range(2,9,2)))
    # ax[2].grid()
    # ax[2].legend(loc='upper left')
    plt.legend(prop=labelFont)
    plt.show()
    
def efficiency_analysis():
    def run_struct_pattern(eTree):
        sTree = StructTree(eTree, pattern_method=StructTree.STRUCT_PATTERN)
        sTree.record_boundary(3, 3, 3, 5)
    
    def run_date_pattern(eTree):
        sTree = StructTree(eTree, pattern_method=StructTree.DATE_ANCHOR)
        sTree.record_boundary(1, 3, 3, 5)

    data = pd.DataFrame(columns=['file', 'size', 'struct-pattern', 'date-pattern'])
    fileName = []
    size = []
    structPattern = []
    datePattern = []
    for file in os.listdir(ANNOTATION_PATH):
        if file.split('.')[-1] != 'html':
            continue
        with open(os.path.join(ANNOTATION_PATH, file), encoding='utf-8') as f:
            fileName.append(file)
            eTree = build_lxml_tree(f)
            size.append(len(eTree.xpath('.//*')))
            
            startTime = process_time()
            run_struct_pattern(eTree)
            endTime = process_time()
            structPattern.append(endTime - startTime)
            
            startTime = process_time()
            run_date_pattern(eTree)
            endTime = process_time()
            datePattern.append(endTime - startTime)
    data['file'] = fileName
    data['size'] = size
    data['struct-pattern'] = structPattern
    data['date-pattern'] = datePattern
    data.set_index('file', drop=True, inplace=True)
    data.to_csv('size-vs-time.csv')

def anchor_precision():
    for method in [StructTree.TAG_ATTRIB_PATTERN, StructTree.HTP_PATTERN, StructTree.STRUCT_PATTERN]:
        precision = []
        for file in os.listdir(ANNOTATION_PATH):
            if file.split('.')[-1] != 'html':
                continue
            with open(os.path.join(ANNOTATION_PATH, file), encoding='utf-8') as f:
                eTree = build_lxml_tree(f)
                nfile = NAGATIVE_PATH + file.split('_')[1]
                if os.path.exists(nfile):
                    neTree = build_lxml_tree(open(nfile, encoding='utf-8'))
                    body = neTree.xpath('//body')
                    if len(body) > 0:
                        body = body[0]
                        eTree.append(body)
                sTree = StructTree(eTree, pattern_method=method)
                frequentPattern = sTree.surffixTree.frequent_pattern(3, 3)
                if len(frequentPattern) == 0:
                    continue
                for pattern, _ in sorted(list(frequentPattern.items()), key=lambda x:-len(x[1])):
                    patternIndexes = sTree._pattern_reduction(frequentPattern[pattern])
                    cnt = 0
                    for i in patternIndexes:
                        p = sTree[i[0]]
                        while p:
                            if "data-record-boundary" in p.elm.attrib:
                                cnt += 1
                                break
                            p = p.parent
                    if cnt == 0:
                        continue
                    precision.append(cnt/len(patternIndexes))
                    break
        precision = sum(precision) / len(precision)
        print(f"{method}: {precision:.3f}")

def anchor_recall():
    def calculate_recall(node: StructNode, anchorIndexes, hit):
        ret = node.index in anchorIndexes
        for child in node.children:
            if calculate_recall(child, anchorIndexes, hit):
                ret = True
        if 'data-record-boundary' in node.attrib:
            if ret:
                hit.add(node.index)
            return False
        else:
            return ret

    for method in [StructTree.TAG_ATTRIB_PATTERN, StructTree.HTP_PATTERN, StructTree.STRUCT_PATTERN]:
        recall = []
        for file in os.listdir(ANNOTATION_PATH):
            if file.split('.')[-1] != 'html':
                continue
            with open(os.path.join(ANNOTATION_PATH, file), encoding='utf-8') as f:
                eTree = build_lxml_tree(f)
                sTree = StructTree(eTree, pattern_method=method)
                frequentPattern = sTree.surffixTree.frequent_pattern(3, 3)
                r = 0
                truth = sTree.elm.xpath('//*[@data-record-boundary]')
                if len(truth) == 0:
                    print(f"no label in {file}.")
                    continue
                
                for pattern, _ in sorted(list(frequentPattern.items()), key=lambda x:-len(x[1])):
                    patternIndexes = sTree._pattern_reduction(frequentPattern[pattern])
                    anchorIndexes = set(sTree._get_anchor(patternIndexes))
                    hit = set()
                    calculate_recall(sTree.root, anchorIndexes, hit)
                    r = max(r, len(hit)/len(truth))
                    if len(hit) == len(truth):
                        break
                
                recall.append(r)
        recall = sum(recall) / len(recall)
        print(f"{method}: {recall:.3f}")

def label_tbdw():
    path = os.path.join('data', 'tbdw')
    with open(os.path.join(path, 'xpath.json')) as f:
        xpath = json.load(f)
    for dir in os.listdir(path):
        if not os.path.isdir(os.path.join(path, dir)):
            continue
        if dir != '50':
            continue
        for file in os.listdir(os.path.join(path, dir)):
            if file[-5:] != '.html':
                continue
            f = open(os.path.join(path, dir, file), encoding='utf-8', errors='ignore')
            root = etree.fromstring(f.read(), parser=etree.HTMLParser())
            f.close()
            for elm in root.xpath(xpath[dir]):
                elm.attrib['data-record-boundary'] = '1'
            f = open(os.path.join(path, dir, file), 'wb')
            f.write(etree.tostring(root, method='html', encoding='utf-8'))
            f.close()

def generate_noise():
    from synthesize import random_tree
    root = etree.Element('html')
    for _ in range(100):
        record = random_tree(50).to_etree()
        record.attrib['data-record-boundary'] = '1'
        root.append(record)
    with open(join('data', 'random', '1.html'), 'wb') as f:
        f.write(etree.tostring(root, method='html', encoding='utf-8'))

def distance_vs_recall():
    def etree2zss(etreeNode):
        zssNode = Node(etreeNode.tag)
        zssNode.cnt = 1
        for child in etreeNode:
            if not hasattr(child, "tag") or type(child.tag) != str or child.tag in ['script', 'noscript', 'style']:
                continue
            childNode = etree2zss(child)
            zssNode.addkid(childNode)
            zssNode.cnt += childNode.cnt
        return zssNode

    def record_distance(r1, r2):
        t1 = etree2zss(r1)
        t2 = etree2zss(r2)
        d = simple_distance(t1, t2)
        return d/(max(t1.cnt, t2.cnt))

    def build_distance_graph(path):
        print(f"Building distance graph in the folder {path}.")
        record = []
        for f in os.listdir(path):
            if f.split('.')[-1] != 'html':
                continue
            with open(os.path.join(path, f), encoding='utf-8', errors='ignore') as f:
                root = html.fromstring(f.read())
                r = root.xpath('//*[@data-record-boundary]')
                if len(r) == 0:
                    print(f"{f} has 0 records!")
                    continue
                # r = random.sample(r, 1)
                record.extend(r)
        
        output = []
        for i in range(len(record)):
            for j in range(i + 1, len(record)):
                output.append((i, j, record_distance(record[i], record[j])))
            print(f"Finished {(i + 1)/len(record):.3f} of the graph in folder {path}.")
        output.sort(key=lambda x:x[-1])
        return output
    
    def largest_component_size(graph):
        def find(x, parent):
            if parent[x] != x:
                parent[x] = find(parent[x], parent)
            return parent[x]
        
        def union(x1, x2, parent, size):
            x1 = find(x1, parent)
            x2 = find(x2, parent)
            if x1 != x2:
                if size[x1] < size[x2]:
                    x1, x2 = x2, x1
                parent[x2] = x1
                size[x1] += size[x2]
        node = set([x[0] for x in graph] + [x[1] for x in graph])
        parent = {x:x for x in node}
        size = {x:1 for x in node}
        for x1, x2, _ in graph:
            union(x1, x2, parent, size)
        maxSize = 1
        for x in node:
            maxSize = max(maxSize, size[find(x, parent)])
        return maxSize

    def threshold_vs_recall(dataPath, threshold):
        graphPath = os.path.join(dataPath, 'graph.txt')
        if os.path.exists(graphPath):# and False:
            graph = []
            with open(graphPath) as f:
                for l in f:
                    l = l.split(' ')
                    graph.append((int(l[0]), int(l[1]), float(l[2])))
        else:
            graph = build_distance_graph(dataPath)
            with open(graphPath, 'w') as f:
                for e in graph:
                    f.write(f"{e[0]} {e[1]} {e[2]:.3f}\n")

        resultPath = os.path.join(dataPath, 'recall.data')
        if os.path.exists(resultPath):# and False:
            recall = []
            with open(resultPath) as f:
                for l in f:
                    recall.append(float(l))
        else:
            nodeCnt = len(set([x[0] for x in graph] + [x[1] for x in graph]))
            recall = []
            for sth in threshold:
                print(f"Computing recall for sth = {sth} in folder {dataPath}.")
                subgraph = graph[:bisect([e[-1] for e in graph], sth)]
                recall.append(largest_component_size(subgraph) / nodeCnt)
            with open(resultPath, 'w') as f:
                for r in recall:
                    f.write(f"{r:.3f}\n")
        return recall

    modernPath = os.path.join('data', 'modern')
    tbdwPath = os.path.join('data', 'tbdw')
    randomPath = join('data', 'random')
    resolution = 100
    threshold = [x/resolution for x in range(0, resolution + 1)]
    fig, ax = plt.subplots(1, 1)
    for dir in os.listdir(modernPath):
        if not os.path.isdir(join(modernPath, dir)): continue
        recall = threshold_vs_recall(join(modernPath, dir), threshold)
        ax.plot(threshold, recall, label=dir)

    for dir in ['1', '25', '50']:
        recall = threshold_vs_recall(join(tbdwPath, dir), threshold)
        ax.plot(threshold, recall, label='tbdw-' + dir)
        
    recall = threshold_vs_recall(randomPath, threshold)
    ax.plot(threshold, recall, label='random')

    ax.set_xlabel('similarity threshold')
    ax.set_ylabel('recall')
    ax.set_ylim([0, 1.03])
    plt.legend(loc="lower right")
    plt.legend()
    plt.show()

def signal_of_old_vs_new():
    bbox = dict(boxstyle ="round",facecolor="white")
    fig, axes = plt.subplots(3,2)
    files = [['tbdw-1', 'amazon'], ['tbdw-25','google'], ['tbdw-50', 'comments']]
    for i, group in enumerate(files):
        for j, file in enumerate(group):
            with open(join('signal-old-vs-new', file + '.html'), encoding='utf-8') as f:
                eTree = build_lxml_tree(f)
                sTree = StructTree(eTree, pattern_method=StructTree.HTP_PATTERN)
                axes[i][j].plot(sTree.nodeEncodingSequence, label=file, marker='.', markersize='2', linewidth='0.5', color='black')
                #axes[i][j].set_xlabel('Sequence Position')
                #axes[i][j].set_ylabel('Tag Path Code Value')
                axes[i][j].tick_params(labelsize=10)
                # axes[i][j].set_yticks(fontsize=10)
                axes[i][j].annotate(file, xy=(0,0), xytext=(0.1, 0.7), xycoords='axes fraction', bbox=bbox)
    plt.annotate('  ----------Sequence Position---------->', xy=(0,0), xytext=(0.02,0.02), xycoords='figure fraction')
    plt.annotate('  ----------Tag Path Code Value---------->', xy=(0,0), xytext=(0.02,0.02), xycoords='figure fraction', rotation=90)
    plt.show()
    """
    top=0.995,
    bottom=0.12,
    left=0.1,
    right=0.995,
    hspace=0.245,
    wspace=0.16
    """

def test_google(method=StructTree.STRUCT_PATTERN):
    print(f"Testing {method} on Google...")
    inputFolder = join('data', 'google')
    outputFolder = join('data', 'google-result')
    for file in os.listdir(outputFolder):
        os.remove(os.path.join(outputFolder, file))
    hit1, miss1, mistake1 = 0, 0, 0
    hit2, miss2, mistake2 = 0, 0, 0
    files = os.listdir(inputFolder)
    progress = 10
    for i, f in enumerate(os.listdir(inputFolder)):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if f.split('.')[-1] != 'html':
            continue
        # f = '2.html'
        with open(join(inputFolder, f), encoding='utf-8') as file:
            eTree = build_lxml_tree(file.read())
        sTree = StructTree(eTree, pattern_method=method)
        xpath = "//div[.//a[./h3 and ./div/cite] and (following-sibling::div[.//a[./h3 and ./div/cite]] or preceding-sibling::div[.//a[./h3 and ./div/cite]]) and not(.//div[@data-q]) and not(.//div[.//a[./h3 and ./div/cite] and (following-sibling::div[.//a[./h3 and ./div/cite]] or preceding-sibling::div[.//a[./h3 and ./div/cite]])])]"
        xpath = "//div[./div/div/div/a[./h3 and ./div/cite]]"
        hiti, missi, mistakei = score(sTree, xpath, 3, 5, 5, 5)
        if int(f.split('.')[0]) % 2 == 0:
            hit1 += hiti
            miss1 += missi
            mistake1 += mistakei
        else:
            hit2 += hiti
            miss2 += missi
            mistake2 += mistakei
        if outputFolder is not None:
            with open(os.path.join(outputFolder, f), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
        
    print(f"Evaluation on Google:\n\t{hit1=}, {miss1=}, {mistake1=}\n{hit2=}, {miss2=}, {mistake2=}") 
    # hit1=1515, miss1=198, mistake1=222 recall=0.911 precision=0.886
    # hit2=1442, miss2=0, mistake2=8 recall=1 precision=1

def test_google_MiBAT():
    print("Testing MiBAT on Google...")
    inputFolder = join('data', 'google')
    outputFolder = join('data', 'google-result')
    for file in os.listdir(outputFolder):
        os.remove(os.path.join(outputFolder, file))
    hit1, miss1, mistake1 = 0, 0, 0
    hit2, miss2, mistake2 = 0, 0, 0
    files = os.listdir(inputFolder)
    progress = 10
    for i, f in enumerate(os.listdir(inputFolder)):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if f.split('.')[-1] != 'html':
            continue
        with open(join(inputFolder, f), encoding='utf-8') as file:
            eTree = build_lxml_tree(file.read())
        anchors = eTree.xpath('//h3')
        sTree = StructTree(eTree, pattern_method=StructTree.MiBAT_ANCHOR, MiBAT_ANCHORS=anchors)
        xpath = "//div[.//a[./h3 and ./div/cite] and (following-sibling::div[.//a[./h3 and ./div/cite]] or preceding-sibling::div[.//a[./h3 and ./div/cite]]) and not(.//div[@data-q]) and not(.//div[.//a[./h3 and ./div/cite] and (following-sibling::div[.//a[./h3 and ./div/cite]] or preceding-sibling::div[.//a[./h3 and ./div/cite]])])]"
        xpath = "//div[./div/div/div/a[./h3 and ./div/cite]]"
        hiti, missi, mistakei = score(sTree, xpath, 1, 5, 5, 5)
        if int(f.split('.')[0]) % 2 == 0:
            hit1 += hiti
            miss1 += missi
            mistake1 += mistakei
        else:
            hit2 += hiti
            miss2 += missi
            mistake2 += mistakei
        if outputFolder is not None:
            with open(os.path.join(outputFolder, f), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
        
    print(f"Evaluation on Google:\n\t{hit1=}, {miss1=}, {mistake1=}\n{hit2=}, {miss2=}, {mistake2=}") 

def test_amazon(method=StructTree.STRUCT_PATTERN):
    print(f"Testing {method} on Amazon...")
    inputFolder = join('data', 'amazon')
    outputFolder = join('data', 'amazon-result')
    for file in os.listdir(outputFolder):
        os.remove(os.path.join(outputFolder, file))
    hit, miss, mistake = 0, 0, 0
    files = os.listdir(inputFolder)
    progress = 10
    for i, f in enumerate(os.listdir(inputFolder)):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if f.split('.')[-1] != 'html':
            continue
        with open(join(inputFolder, f), encoding='utf-8') as file:
            eTree = build_lxml_tree(file.read())
        sTree = StructTree(eTree, pattern_method=method)
        xpath = '//div[@data-asin and @data-uuid and .//span[@class="a-icon-alt" or @class="a-price-whole"] and .//span[@data-component-type="s-product-image"] and not(.//div[@data-uuid and .//span[@class="a-icon-alt" or @class="a-price-whole"] and .//span[@data-component-type="s-product-image"]])]'

        hiti, missi, mistakei = score(sTree, xpath, 3, 5, 5, 5)
        hit += hiti
        miss += missi
        mistake += mistakei
        if outputFolder is not None:
            with open(os.path.join(outputFolder, f), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
        
    print(f"Evaluation on Amazon:\n\t{hit=}, {miss=}, {mistake=}") 
    # hit=4602, miss=232, mistake=257, recall=0.9931 precision=0.935

def test_amazon_MiBAT():
    print("Testing MiBAT on Amazon...")
    inputFolder = join('data', 'amazon')
    outputFolder = join('data', 'amazon-result')
    for file in os.listdir(outputFolder):
        os.remove(os.path.join(outputFolder, file))
    hit, miss, mistake = 0, 0, 0
    files = os.listdir(inputFolder)
    progress = 10
    for i, f in enumerate(os.listdir(inputFolder)):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if f.split('.')[-1] != 'html':
            continue
        with open(join(inputFolder, f), encoding='utf-8') as file:
            eTree = build_lxml_tree(file.read())
        anchors = eTree.xpath('//span[@class="a-price-whole"]')
        sTree = StructTree(eTree, pattern_method=StructTree.MiBAT_ANCHOR, MiBAT_ANCHORS=anchors)
        xpath = '//div[@data-asin and @data-uuid and .//span[@class="a-icon-alt" or @class="a-price-whole"] and .//span[@data-component-type="s-product-image"] and not(.//div[@data-uuid and .//span[@class="a-icon-alt" or @class="a-price-whole"] and .//span[@data-component-type="s-product-image"]])]'

        hiti, missi, mistakei = score(sTree, xpath, 1, 5, 5, 5)
        hit += hiti
        miss += missi
        mistake += mistakei
        if outputFolder is not None:
            with open(os.path.join(outputFolder, f), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
        
    print(f"Evaluation on Amazon:\n\t{hit=}, {miss=}, {mistake=}") 
    # hit=4587, miss=247, mistake=333, recall=0.945 precision=0.932

def test_comment(method=StructTree.STRUCT_PATTERN, onePerSite=True):
    print(f"Testing {method} on Comment...")
    inputFolder = join('data', 'comment')
    outputFolder = join('data', 'comment-result')
    for file in os.listdir(outputFolder):
        os.remove(os.path.join(outputFolder, file))
    hit, miss, mistake = 0, 0, 0
    sites = set()
    missCnt = {}
    mistakeCnt = {}
    files = os.listdir(inputFolder)
    progress = 10
    for i, f in enumerate(os.listdir(inputFolder)):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if f.split('.')[-1] != 'html':
            continue
        #f = 'ann_73-1.html'
        site = f.split('_')[1].split('-')[0]
        if onePerSite :
            if site in sites:
                continue
            else:
                sites.add(site)
        with open(join(inputFolder, f), encoding='utf-8') as file:
            eTree = build_lxml_tree(file.read())
        sTree = StructTree(eTree, pattern_method=method)
        xpath = '//*[@data-record-boundary]'

        hiti, missi, mistakei = score(sTree, xpath, 3, 5, 2, 3)
        hit += hiti
        miss += missi
        mistake += mistakei
        missCnt[site] = missCnt.get(site, 0) + missi
        mistakeCnt[site] = mistakeCnt.get(site, 0) + mistakei
        if outputFolder is not None:
            with open(os.path.join(outputFolder, f), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
    print('Misses:\n')
    print(sorted([(k, v) for k, v in missCnt.items()], reverse=True, key=lambda x:x[::-1]))
    print('Mistakes:\n')
    print(sorted([(k, v) for k, v in mistakeCnt.items()], reverse=True, key=lambda x:x[::-1]))
    print(f"Evaluation on Comment:\n\t{hit=}, {miss=}, {mistake=}") 
    # hit=55608, miss=2498, mistake=3067 recall=0.957 precision=0.948

def test_comment_MiBAT(onePerSite=True):
    print("Testing MiBAT on Comment...")
    inputFolder = join('data', 'comment')
    outputFolder = join('data', 'comment-result')
    for file in os.listdir(outputFolder):
        os.remove(os.path.join(outputFolder, file))
    hit, miss, mistake = 0, 0, 0
    sites = set()
    missCnt = {}
    mistakeCnt = {}
    files = os.listdir(inputFolder)
    progress = 10
    for i, f in enumerate(os.listdir(inputFolder)):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if f.split('.')[-1] != 'html':
            continue
        # f = 'ann_73-1.html'
        site = f.split('_')[1].split('-')[0]
        if onePerSite :
            if site in sites:
                continue
            else:
                sites.add(site)
        with open(join(inputFolder, f), encoding='utf-8') as file:
            eTree = build_lxml_tree(file.read())
        sTree = StructTree(eTree, pattern_method=StructTree.MiBAT_ANCHOR)
        xpath = '//*[@data-record-boundary]'

        hiti, missi, mistakei = score(sTree, xpath, 1, 5, 2, 3)
        hit += hiti
        miss += missi
        mistake += mistakei
        missCnt[site] = missCnt.get(site, 0) + missi
        mistakeCnt[site] = mistakeCnt.get(site, 0) + mistakei
        if outputFolder is not None:
            with open(os.path.join(outputFolder, f), 'wb') as of:
                of.write(html.tostring(eTree, pretty_print=True))
    print('Misses:\n')
    print(sorted([(k, v) for k, v in missCnt.items()], reverse=True, key=lambda x:x[::-1]))
    print('Mistakes:\n')
    print(sorted([(k, v) for k, v in mistakeCnt.items()], reverse=True, key=lambda x:x[::-1]))
    print(f"Evaluation on Comment:\n\t{hit=}, {miss=}, {mistake=}")
    # hit=1610, miss=829, mistake=18

def elm2text(e):
    textElm = [x for x in e.xpath('descendant-or-self::*[@data-index]') if x.text is not None] # only consider elements that are touched
    return ' '.join([x.text.strip() for x in textElm]).strip()

def score(sTree: StructTree, annotationXpath, lenThresh: int, freqThresh: int, recordHeightThresh: int, recordSizeThresh: int):
    annotatedElement = sTree.root.elm.xpath(annotationXpath)
    index2text = {}
    annotatedTexts = set()
    for e in annotatedElement:
        t = elm2text(e)
        index2text[int(e.attrib['data-index'])] = t
        annotatedTexts.add(t)
    
    recordGroups = sTree.record_boundary(lenThresh, freqThresh, recordHeightThresh, recordSizeThresh)
    for recordIndexes in recordGroups:
        for i in recordIndexes:
            if i not in index2text:
                t = elm2text(sTree[i].elm)
                index2text[i] = t

    # only evaluate records that found with ground truth
    detectedIndexes = set()
    for recordIndexes in recordGroups:
        groupTexts = set([elm2text(sTree[i].elm) for i in recordIndexes])
        if len(groupTexts.intersection(annotatedTexts)) > 0:
            detectedIndexes.update(set(recordIndexes))
    
    hit = set()
    miss = set()
    mistake = set()
    detectedTexts = set()
    for i in detectedIndexes:
        t = index2text[i]
        detectedTexts.add(t)
        if len(t) == 0:
            continue
        if t in annotatedTexts:
            sTree[i].elm.attrib["style"] = "border: thick solid darkgreen;"
            hit.add(t)
        else:
            sTree[i].elm.attrib["style"] = "border: thick solid yellow;"
            mistake.add(t)
    for e in annotatedElement:
        t = index2text[int(e.attrib['data-index'])]
        if len(t) == 0:
            continue
        if t not in detectedTexts:
            e.attrib["style"] = "border: thick solid red;"
            miss.add(t)
    return len(hit), len(miss), len(mistake)

def record_largest_string(element, ans, existingKeys):
    if len(ans) > 0 and 'data-record-boundary' in element.attrib:
        return ans
    if element.text is not None:
        text = ' '.join(element.text.split())
        if text not in existingKeys and len(text) > len(ans):
            ans = text
    for child in element:
        ans = record_largest_string(child, ans, existingKeys)
    return ans

def text_score(targetFile, resultFile):
    with open(targetFile, encoding='utf-8') as f:
        tree = etree.fromstring(f.read(), parser=etree.HTMLParser())
    recordKeys = set()
    for element in tree.xpath('//*[@data-record-boundary]'):
        key = record_largest_string(element, '', recordKeys)
        if len(key) > 0:
            recordKeys.add(key)

    with open(resultFile, encoding='utf-8') as f:
        recordRegions = f.read().strip().split('\n\n')
    if len(recordRegions) == 1 and len(recordRegions[0]) == 0:
        print(f'Skipping {targetFile} that has no records.')
        return -1, -1, -1
    for i, region in enumerate(recordRegions):
        region = region.split('\n')
        for j, record in enumerate(region):
            record = set(record.split('|'))
            region[j] = record
        recordRegions[i] = region
    targetRegions = []
    for region in recordRegions:
        matchCnt = 0
        for record in region:
            if len(recordKeys.intersection(record)) > 0:
                matchCnt += 1
            if matchCnt > 2:
                targetRegions.append(region)
                break
    
    detectedRecords = [record for region in targetRegions for record in region]
    # hit = set()
    hitCnt = 0
    for record in detectedRecords:
        if len(record.intersection(recordKeys)) > 0:
            hitCnt += 1
    hitCnt = min(hitCnt, len(recordKeys))
    mistakeCnt = len(detectedRecords) - hitCnt
    missCnt = len(recordKeys) - hitCnt
    assert mistakeCnt >= 0 and missCnt >= 0
    return hitCnt, missCnt, mistakeCnt

def evaluate_record_texts(inputFolder, resultFolder, filterFunction = None):
    print(f'Evaluating {inputFolder} against {resultFolder}...')
    hit = miss = mistake = 0
    progress = 10
    files = os.listdir(inputFolder)
    skipCnt = 0
    for i, file in enumerate(files):
        if int((i+1)/len(files) * 100) == progress:
            print(f'Finished {progress}%.')
            progress += 10
        if file.split('.')[-1] != 'html':
            continue
        if filterFunction and filterFunction(file):
            continue
        hiti, missi, mistakei = text_score(join(inputFolder, file), join(resultFolder, file.split('.')[0] + '.txt'))
        if hiti < 0:
            skipCnt += 1
            continue
        hit += hiti
        miss += missi
        mistake += mistakei
    print(f'{hit=}, {miss=}, {mistake=}\nSkipped {skipCnt} of {len(files)} files.')

def size_depth_statistics(folder):
    def dfs(node, depth, depthList):
        assert node.tag is not None
        size = 1
        if len(node) == 0:
            depthList.append(depth)
        else:
            for child in node:
                size += dfs(child, depth + 1, depthList)
        return size
    
    depthList, sizeList = [], []
    
    cleaner = Cleaner(remove_tags=('br',))
    for file in os.listdir(folder):
        if os.path.isdir(join(folder, file)):
            sizeListi, depthListi = size_depth_statistics(join(folder, file))
            sizeList.extend(sizeListi)
            depthList.extend(depthListi)
            continue
        if file.split('.')[-1] != 'html':
            continue
        with open(join(folder, file), encoding='utf-8') as f:
            tree = etree.fromstring(cleaner.clean_html(f.read()), parser=etree.HTMLParser())
            sizeList.append(dfs(tree, 0, depthList))
    
    return sizeList, depthList

def record_statistics(folder):
    cntList = []
    
    cleaner = Cleaner(remove_tags=('br',), safe_attrs_only=False)
    for file in os.listdir(folder):
        if os.path.isdir(join(folder, file)):
            cnti = record_statistics(join(folder, file))
            cntList.extend(cnti)
            continue
        if file.split('.')[-1] != 'html':
            continue
        with open(join(folder, file), encoding='utf-8') as f:
            tree = etree.fromstring(cleaner.clean_html(f.read()), parser=etree.HTMLParser())
            cntList.append(len(tree.xpath('//*[@data-record-boundary]')))
    
    return cntList

def dom_statistics():
    for dataset in ['tbdw', 'amazon', 'google', 'comment']:
        # evaluate_record_texts(join('data', dataset), join('data', f'{dataset}-result-signal'))
        domStats = False
        recordStats = True
        if domStats:
            sizeList, depthList = dom_statistics(join('data', dataset))
            avgSize = statistics.mean(sizeList)
            stdSize = statistics.stdev(sizeList)
            avgDepth = statistics.mean(depthList)
            stdDepth = statistics.stdev(depthList)
            print(f"{dataset}: {avgSize=}, {stdSize=}, {avgDepth=}, {stdDepth=}")
        if recordStats:
            cntList = record_statistics(join('data', dataset))
            avgCnt = statistics.mean(cntList)
            stdCnt = statistics.stdev(cntList)
            print(f"{dataset}: {avgCnt=}, {stdCnt=}")

if __name__ == '__main__':
    plot_efficiency()
    exit(0)