from importlib.resources import path
import json
from operator import index
from pydoc import pathdirs
from sys import intern
from typing import Dict, List, Tuple, Set
from lxml import html
from json.decoder import JSONDecodeError
import matplotlib.pyplot as plt
import os, re
from types import SimpleNamespace
from collections import defaultdict, deque, Counter
import functools
from suffix_tree import Tree
from suffix_tree.node import Node, Internal, Leaf
import dateparser
from bisect import bisect, bisect_left, bisect_right

TAG_BLACK_LIST = ['script', 'noscript', 'head', 'meta', 'style']
ATTRIB_BLACK_LIST = ['data-record-boundary']
def build_lxml_tree(input):
    if hasattr(input, 'read'):
        input = input.read()
    try:
        # if the input is json type, we will convert the json into an etree
        def json_build(parent, jsonNode):
            if type(jsonNode) == dict:
                child = html.Element('dict')
                for key, val in jsonNode.items():
                    grandChild = html.Element(key)
                    child.append(grandChild)
                    if type(val) == str:
                        grandChild.text = val
                    else:
                        json_build(grandChild, val)
            elif type(jsonNode) == list:
                child = html.Element('list')
                for n in jsonNode:
                    json_build(child, n)
            else:
                child = html.Element('null')
                child.text = str(jsonNode)
            parent.append(child)

        input = json.loads(input)
        root = html.Element('json')
        json_build(root, input)
    except JSONDecodeError as e:
        # otherwise, we simply build an etree from the HTML/XML source
        root = html.fromstring(input)
    return root

class SuffixTree(Tree):
    def __init__(self, s=None):
        super().__init__({1:s})
        self._count_leaf()
        
    def _count_leaf(self):
        def count(node: Node):
            if node.is_internal():
                node.leafCnt = 0
                for child in node.children.values():
                    node.leafCnt += child.leafCnt
            else:
                node.leafCnt = 1
        
        self.root.post_order(count)
    
    def frequent_pattern(self, freqThresh, lenThresh):
        def reduce_leaf_cnt(node: Internal, root: Internal):
            for child in node.children.values():
                if child.leafCnt >= freqThresh:
                    node.leafCnt -= child.leafCnt
                    reduce_leaf_cnt(child, root)
            for c in node.path.S[node.path.start:node.path.end]:
                # reduce the frequency count of the sub-suffix because we only return the patterns closed at the highest frequency that satisfy both the length and frequency thresholds.
                root.children[c].leafCnt -= node.leafCnt
        
        def get_suffix_indexes(node: Node, ans:List[int]):
            if node.is_leaf():
                ans.append(node.path.start)
            else:
                for child in node.children.values():
                    get_suffix_indexes(child, ans)

        def search(node: Internal, root: Internal, freqThresh, lenThresh, ans:Dict[str, List[Tuple[int, int]]]):
            if node.is_leaf() or node.leafCnt < freqThresh: return
            pathLen = node.path.end - node.path.start
            if pathLen >= lenThresh:
                pattern = node.path.S[node.path.start:node.path.end]
                indexes = []
                get_suffix_indexes(node, indexes)
                reduce_leaf_cnt(node, root)
                # ans[pattern] = [(i[0], i[0] + lenThresh) for i in indexes]
                ans[pattern] = [(i, i + len(pattern)) for i in indexes]
            else:
                # for child in sorted(node.children.values(), key=lambda x:-x.leafCnt):
                for child in node.children.values():
                    search(child, root, freqThresh, lenThresh, ans)
        ans = {}
        search(self.root, self.root, freqThresh, lenThresh, ans)
        return ans

class StructNode:
    def __init__(self, elm, depth, parent: 'StructNode', root: 'StructTree') -> None:
        self.elm = elm
        self.parent = parent
        self.root = root
        self.tag = elm.tag
        self.tagAttrib = (elm.tag, tuple(sorted([k for k in elm.keys() if k not in ATTRIB_BLACK_LIST])))
        self.tagAttribID: int = None
        self.tagID: int = None
        self.htpID: int = None
        self.structure = None
        self.structID: int = None
        self.size = 1
        self.height = 1
        self.depth = depth
        # self.attribCnt = 0
        self.children = []
        self.startIndex = len(root.nodeSequence)
        for childElement in elm:
            if not (hasattr(childElement, 'tag') and type(childElement.tag) == str) or childElement.tag in TAG_BLACK_LIST:
                continue
            childStructNode = StructNode(childElement, depth + 1, self, root)
            self.children.append(childStructNode)
            self.size += childStructNode.size
            self.height = max(self.height, childStructNode.height + 1)
            # self.attribCnt += childStructNode.attribCnt
        # self.attribCnt += len([a for a in elm.attrib if a not in ATTRIB_BLACK_LIST])
        self.index = len(root.nodeSequence)
        root.nodeSequence.append(self)
        self.endIndex = self.index + 1
        self.attrib = elm.attrib
        elm.attrib['data-height'] = str(self.height)
        elm.attrib['data-size'] = str(self.size)
        elm.attrib['data-index'] = str(self.index)
        elm.attrib['data-depth'] = str(self.depth)
    
    @functools.cached_property
    def ancestor_indexes(self):
        if self.parent is None:
            return (self.index,)
        return self.parent.ancestor_indexes + (self.index,)

class StructTree(StructNode):
    TAG_ATTRIB_PATTERN = 'TAG_ATTRIBUTE_ENCODING'
    HTP_PATTERN = 'HTP_ENCODING'
    STRUCT_PATTERN = 'STRUCTURE_ENCODING'
    MiBAT_ANCHOR = 'MiBAT_ANCHOR'
    URL_ANCHOR = 'URL_ANCHOR'
    def __init__(self, elm, pattern_method = STRUCT_PATTERN, MiBAT_ANCHORS=None) -> None:
        self.tagAttrib2ID: Dict[tuple, int] = {} # node signature -> node signature ID
        self.tag2ID: Dict[tuple, int] = {} # tag -> tag ID
        self.struct2ID: Dict[tuple, int] = {} # structure signature -> structure ID
        self.htp2ID: Dict[tuple, int] = {}

        self.structID2Index: Dict[int, List[int]] = {} # structure ID -> list of node indexes of the structure
        self.tagAttribID2Index: Dict[int, List[int]] = {}
        self.htpID2Index: Dict[int, List[int]] = {}

        self.structFreqency: Dict[int, int] = {} # structure frequency
        # self.htpFrequency: Dict[int, int] = {}
        # self.tagAttribFrequency: Dict[int, int] = {}
        self.structSize: Dict[int, int] = {} # structure ID -> size of the structure
        self.structHeight: Dict[int, int] = {} # structure ID -> height of the structure
        self.nodeSequence: List[StructNode] = []
        super().__init__(elm, 0, None, self)
        self._assign_ID()
        self.patternMethod = pattern_method
        
        if pattern_method == StructTree.TAG_ATTRIB_PATTERN:
            # self.nodeEncodingSequence = tuple([x.tagAttribID for x in self.nodeSequence])
            self.nodeEncodingSequence = tuple([x.tagID for x in self.nodeSequence])
        elif pattern_method == StructTree.HTP_PATTERN:
            self.nodeEncodingSequence = tuple([x.htpID for x in self.nodeSequence])
        elif pattern_method == StructTree.STRUCT_PATTERN:
            self.nodeEncodingSequence = tuple([x.structID for x in self.nodeSequence])
        elif pattern_method == StructTree.MiBAT_ANCHOR:
            # assert MiBAT_ANCHORS is not None, "MiBAT_ANCHORS is required when using the MiBAT method."
            self.nodeEncodingSequence = self._mibat_anchor_encoding(MiBAT_ANCHORS)
        elif pattern_method == StructTree.URL_ANCHOR:
            self.nodeEncodingSequence = self._url_anchor_encoding()
        else:
            raise ValueError
        # self.codesFrequency = Counter(self.nodeEncodingSequence)
        self.surffixTree = SuffixTree(self.nodeEncodingSequence)

    def __getitem__(self, i) -> StructNode:
        return self.nodeSequence[i]

    def _date_string_signature(self, strList):
        IS_NUMBER = 1
        IS_ALPHA = 2
        ret = []
        for s in strList:
            i = 0
            signature = []
            while i < len(s):
                if s[i].isdigit():
                    j = i + 1
                    while j < len(s):
                        if s[j].isdigit():
                            j += 1
                        else:
                            break
                    i = j
                    signature.append(IS_NUMBER)
                elif s[i].isalpha():
                    j = i + 1
                    while j < len(s):
                        if s[j].isalpha():
                            j += 1
                        else:
                            break
                    i = j
                    signature.append(IS_ALPHA)
                else:
                    j = i + 1
                    while j < len(s):
                        if not s[j].isdigit() and not s[j].isalpha():
                            j += 1
                        else:
                            break
                    i = j
                    signature.append(0)
            ret.append(tuple(signature))
        return tuple(ret)
    
    def _mibat_anchor_encoding(self, MiBAT_ANCHORS = None, minDateStringLen = 2, maxDateStringLen = 30):
        patternID = {}
        codes = list(range(-1, -len(self.nodeSequence) - 1, -1))
        MiBAT_ANCHORS = set([int(e.attrib['data-index']) for e in MiBAT_ANCHORS])
        for i, node in enumerate(self.nodeSequence):
            if MiBAT_ANCHORS:
                if node.index in MiBAT_ANCHORS:
                    codes[i] = 1
                continue
            if node.size != 1:
                continue
            s = node.elm.text
            if s is None or len(s) < minDateStringLen: 
                continue
            s = ' '.join(s.split())
            if s.isdecimal():
                continue
            if len(s) < minDateStringLen or len(s) > maxDateStringLen:
                continue
            if dateparser.parse(s) is None:
                continue
            codes[i] = patternID.setdefault(self._date_string_signature(re.split('-/,: ', s)), len(patternID) + 1)
        
        return codes
    
    def _url_anchor_encoding(self):
        patternID = {}
        codes = list(range(-1, -len(self.nodeSequence) - 1, -1))
        for i, node in enumerate(self.nodeSequence):
            if node.tag != 'a':
                continue
            pattern = tuple([x.tag for x in node.parent.children])
            codes[i] = patternID.setdefault(pattern, len(patternID) + 1)
        return codes
    
    def _assign_ID(self):
        """
        Assign node tagAttrib ID, HTP ID, and structure ID for each node
        """
        def dfs(node: StructNode, htp):
            node.tagAttribID = self.tagAttrib2ID.setdefault(node.tagAttrib, len(self.tagAttrib2ID) + 1) # node ID starts from 1
            node.tagID = self.tag2ID.setdefault(node.tag, len(self.tag2ID) + 1)
            # self.tagAttribFrequency[node.tagAttribID] = self.tagAttribFrequency.get(node.tagAttribID, 0) + 1
            htpNew = htp + (node.tagID,)
            node.htpID = self.htp2ID.setdefault(htpNew, len(self.htp2ID) + 1)
            # self.htpFrequency[node.htpID] = self.htpFrequency.get(node.htpID, 0) + 1
            structure = []
            for child in node.children:
                dfs(child, htpNew)
                structure.append(child.structID)
            structure.append(node.tagAttribID)
            node.structure = tuple(structure)
            node.structID = self.struct2ID.setdefault(node.structure, len(self.struct2ID) + 1) # structure ID starts from 1
            self.structID2Index.setdefault(node.structID, []).append(node.index)
            self.structFreqency[node.structID] = self.structFreqency.setdefault(node.structID, 0) + 1
            self.structSize[node.structID] = node.size
            self.structHeight[node.structID] = node.height
        dfs(self, tuple())

    def _pattern_reduction(self, patternIndexes):
        # patternIndexes.sort()
        
        indexes = []
        for li, ri in patternIndexes:
            indexes.extend(list(range(li, ri)))
        lca = self._lowest_common_ancestor(indexes)
        ret = []
        subTreeRange = [(x.startIndex, x.endIndex) for x in self[lca].children]
        for i, pi in enumerate(patternIndexes):
            li = bisect(subTreeRange, (pi[0], pi[0]))
            if li == len(subTreeRange) or (li > 0 and subTreeRange[li][1] > pi[0]):
                li -= 1
            ri = bisect(subTreeRange, (pi[1], pi[1]))
            ans = (-1, -1)
            for j in range(li, ri):
                rangej = subTreeRange[j]
                overlap = (max(rangej[0], pi[0]), min(rangej[1], pi[1]))
                if overlap[1] - overlap[0] > ans[1] - ans[0]:
                    ans = overlap
            if ans != (-1, -1): # a dirty fix of bug that overlap is a single point.
                ret.append(ans)
        return ret
                
        lp, rp = patternIndexes[0]
        if self._lowest_common_ancestor(list(range(lp, rp))) != lca:
            return patternIndexes
        
        dl = 0
        for i in range(len(patternIndexes[0])):
            if self._lowest_common_ancestor([x[0]+i for x in patternIndexes]) != lca:
                break
            dl += 1
        dr = 0
        for i in range(len(patternIndexes[0])):
            # rp is is not included in the pattern!
            if self._lowest_common_ancestor([x[1]-1-i for x in patternIndexes]) != lca:
                break
            rp += 1
        return [(lp+dl, rp-dr) for lp, rp in patternIndexes]

    def _lowest_common_ancestor(self, nodeIndexes: List[int])->int:
        ancestorPaths = [self[i].ancestor_indexes for i in nodeIndexes]
        L = min([len(p) for p in ancestorPaths])
        i = 0
        while i + 1 < L and len(set([path[i+1] for path in ancestorPaths])) == 1:
            i += 1
        return ancestorPaths[0][i]

    def _get_anchor(self, patternIndexes):
        anchorIndexes = set()
        for li, ri in patternIndexes:
            # find the nearest common ancestor of the nodes in the sequence
            anchorIndexes.add(self._lowest_common_ancestor([node.index for node in self.nodeSequence[li:ri]]))
        return sorted(list(anchorIndexes))

    def _align_records(self, anchorIndexes, freqThresh: int):
        class PathTrieNode:
            def __init__(self, children = {}, indexes = [], prevIndexes = []) -> None:
                self.children = children
                self.indexes = indexes
                self.prevIndexes = prevIndexes

        def build_path_trie(indexes, prevIndexes, index2Node:StructTree):
            indexGroups = {}
            prevIndexGroups = {}
            for i in indexes:
                parentNode = index2Node[i].parent
                if parentNode is None: continue
                # indexGroups.setdefault(parentNode.tagAttribID, []).append(parentNode.index)
                # prevIndexGroups.setdefault(parentNode.tagAttribID, []).append(i)
                if self.patternMethod == StructTree.MiBAT_ANCHOR:
                    indexGroups.setdefault(parentNode.htpID, []).append(parentNode.index)
                    prevIndexGroups.setdefault(parentNode.htpID, []).append(i)
                else:
                    indexGroups.setdefault(parentNode.tag, []).append(parentNode.index)
                    prevIndexGroups.setdefault(parentNode.tag, []).append(i)
            children = {}
            for k, v in indexGroups.items():
                children[k] = build_path_trie(v, prevIndexGroups[k], index2Node)
            trieNode = PathTrieNode(children=children, indexes=indexes, prevIndexes=prevIndexes)
            return trieNode
        
        def align(node: PathTrieNode, ans: List[int], index2Node: List[StructNode])->bool:
            # return True if alignment should stop at this trie node
            MAX_MERGE = 2
            if len(node.indexes) < freqThresh:
                return False
            groupCnt = Counter(node.indexes)
            if self._lowest_common_ancestor(groupCnt.keys()) in groupCnt:
                return True
            if max(groupCnt.values()) > MAX_MERGE:
                return True
                if len(groupCnt) == 1 or len(set(groupCnt.values())) > 1:
                    return True
            
            
            # for child in node.children.values():
            #     if align(child, ans, index2Node):
            #         mergedIndex = set(child.indexes)
            #         ans.extend([i for i in node.indexes if index2Node[i].parent.index in mergedIndex])
            # return False
            merge = False
            for child in node.children.values():
                if align(child, ans, index2Node):
                    merge = True
                    ans.extend(child.prevIndexes)
                    
            if merge:
                for child in node.children.values():
                    if len(child.indexes) < freqThresh:
                        ans.extend(child.prevIndexes)

            return False

        root = build_path_trie(anchorIndexes, None, self.nodeSequence)
        ans = []
        if align(root, ans, self):
            return anchorIndexes
        return ans

    def record_boundary(self, lenThresh: int, freqThresh: int, recordHeightThresh, recordSizeThresh):
        # assert freqThresh >= 3

        frequentPattern = self.surffixTree.frequent_pattern(freqThresh, lenThresh)
        recordRegion = set()
        # for sid in freqStruct:
        for pattern, patternIndexesO in frequentPattern.items():
            patternIndexes = self._pattern_reduction(patternIndexesO)
            if patternIndexes[0][1] - patternIndexes[0][0] < lenThresh:
                continue
            anchorIndexes = self._get_anchor(patternIndexes)
            if len(set(anchorIndexes)) < freqThresh:
                continue
            recordContainerIndexes = set(self._align_records(anchorIndexes, freqThresh))
            recordContainerIndexes = [x for x in recordContainerIndexes if self[x].height >= recordHeightThresh and self[x].size >= recordSizeThresh and len(self[x].elm.text_content().split()) > 0]
            if len(recordContainerIndexes) > 0:
                # if all([ "data-record-boundary" not in self[i].elm.attrib for i in recordContainerIndexes]):
                #     continue
                regionNodeIndex = self._lowest_common_ancestor(recordContainerIndexes)
                recordRegion.add(tuple(sorted(recordContainerIndexes)))
        
        ret = set()
        # clean nested trivial records
        # if multiple records has a parent that is also a record, and their parent records are different, then, remove them. If parent records are the same, they could be nested category records.
        recordIndexes = set([i for g in recordRegion for i in g ])
        for group in recordRegion:
            group = set(group)
            parentCnt = 0
            parentRecords= set()
            nestedRecords = set()
            for i in group:
                ancestorIndexes = self[i].ancestor_indexes
                for a in ancestorIndexes[:-1]: # ancestor index includes self
                    if a in recordIndexes and a not in group:
                        parentCnt += 1
                        parentRecords.add(a)
                        nestedRecords.add(i)
                        break
                        # else:
                        #     print('')
            if parentCnt >= len(nestedRecords) * 0.9 and len(parentRecords) >= freqThresh:
                group.difference_update(nestedRecords)
            # else:
            #     print(f'nested category records found. Parents: {parentRecords}, children: {nestedRecords}')
            if(group):
                ret.add(tuple(sorted(group)))
        return ret
    
    def structure_sequence(self, heightThresh: int, sizeThresh: int) -> List[int]:
        return [(i, node.structID) for (i, node) in enumerate (self.nodeSequence) if node.height >= heightThresh and node.size >= sizeThresh]


if __name__ == "__main__":
    input = """
    {
        "key1": "val1",
        "key2": [1, 2, 3, {"key3": "val3"}]
    }
    """
    for file in os.listdir('.'):
        if file.split(".")[-1] != "html":
            continue
        if file.split("-")[-1] == "mod.html":
            continue
        with open(file, encoding='utf-8') as f:
            root = build_lxml_tree(f)
            tree = StructTree(root)
            # with open('foxnews-mod.html', 'wb') as f:
            #     f.write(html.tostring(root))
            # reg = tree.record_region(2, 3, 3)
            (kernelGrowPath, kernel2Record) = tree.record_boundary(2, 2, 10, 3, 5)
            tmp = set(list(kernel2Record.values()))
            # for x in tmp.copy():
            #     node = tree[x].parent
            #     while node:
            #         if node.index in tmp:
            #             tmp.remove(x)
            #             break
            #         node = node.parent
            
            tmp = list(tmp)
            tmp.sort(reverse=True)
            signal = tree.structure_sequence(2, 3)
            signalIndex = [x[0] for x in signal]
            singalVal = [x[1] for x in signal]
            print('showing {}'.format(file))
            # plt.plot(signalIndex, singalVal, 'black')
            for k in tmp:
                node = tree[k]
                node.elm.attrib["style"] = "border: thick solid red;"
            #     sIdx, eIdx = node.startIndex, node.endIndex
            #     plt.plot(range(sIdx, eIdx), [tree[i].structID for i in range(sIdx, eIdx)])
            # plt.show()
            s = html.tostring(root, pretty_print=True)
            with open(file.split('.')[0]+'-mod.html', 'wb') as f:
                f.write(s)
    # comment region problematic: 95-1
    # comment boundary problematic: 18-17, 21-1, 23-7, 95-1, 99-14
    # comment boundary problematic 2: 99-14
    # comment boundary perfect: 43-1, 60-1, 
    # root = build_lxml_tree(input)
    # print(etree.tostring(root, method='xml', pretty_print=True).decode())
