import sys, os
import time
import progressbar
import re
import random
import csv
import argparse 
import importlib.util
import pickle

REMOVE_COMMA  = False
REMOVE_QUOT   = False
PRINT_PROGRESS= False
TRAIN_POS     = False
TRAIN_MARKOV  = False

NOT_PYINSTALLER = True
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'): # this should be set whenever script is run inside pyinstaller bundle
    NOT_PYINSTALLER = False

markov_pos = {}
if not TRAIN_POS:
    path = "src/_pos_model.p" if NOT_PYINSTALLER else os.path.join(sys._MEIPASS, "src/_pos_model.p")
    if os.path.isfile(path):
        markov_pos = pickle.load( open( path, "rb" ) )

markov_chain = {}
if not TRAIN_MARKOV:
    path = "src/_markov_model.p" if NOT_PYINSTALLER else os.path.join(sys._MEIPASS, "src/_markov_model.p")
    if os.path.isfile(path):
        markov_chain = pickle.load( open( path, "rb" ) )

import pl_core_news_sm
nlp = pl_core_news_sm.load()

remove_links = r"\[\d+\]" # useful in wikipedia to remove hyperlinks
sentence_split = re.compile(r"[\.\?;!]")
word_split = re.compile(r"\s+")

def markov_file(filename: str):
    with open(filename) as f:
        if filename.split(".")[-1] == "csv": # added for train.csv, edit it if you want to use your own
            reader = csv.reader(f, quoting=csv.QUOTE_ALL, skipinitialspace=True)
            row_count = sum(1 for r in reader)

            f.seek(0)
            reader = csv.reader(f, quoting=csv.QUOTE_ALL, skipinitialspace=True) # needed to put iterator again at the top of the file

            bar = progressbar.ProgressBar(max_value=row_count,redirect_stdout=True) if PRINT_PROGRESS else (lambda x: x)
            i = 0
            for col in reader:
                markov_it(col[0])
                markov_it(col[1])
                i += 1
                if PRINT_PROGRESS: bar.update(i,force=True)
                #bar.update(force=True)
            return        
        with open(filename) as f:
            data = f.read()
            if PRINT_PROGRESS:
                bar = progressbar.ProgressBar(widgets=[progressbar.SimpleProgress()])
                markov_it(data,bar)
            else:
                markov_it(data)
    
def markov_it(data: str, bar = None):
    data = re.sub(remove_links,"",data)
    data = re.sub(r"Rozdział (?:[XIVL]+)|(?:\w+)\n","",data)
    data = re.sub(r"EPILOG","",data)
    data = re.sub(r"\r?\n","",data)
    data = data.replace("—", "")
    data = data.replace("…", "")
    if REMOVE_COMMA: data = data.replace("," , "")
    if REMOVE_QUOT: data = data.replace("„", "").replace("”", "").replace("\"","")#.replace("\'","") # Might not be needed in most text so i remove it
    sentences = [ x.lstrip() for x in sentence_split.split(data) if x.count(" ") > 1 ]
    # TODO in future add removing parenthesis overall 
    if bar == None or not PRINT_PROGRESS: bar = (lambda x: x) # If progressbar is not specified, then just return passed argument 
    for s in bar(sentences):
        words = word_split.split(s)
        if len(words) <= 1: continue
        for i in range(len(words)-1):
            if not words[i+1]: continue 
            if TRAIN_POS:
                pos_1 = nlp(words[i])[0].pos
                pos_2 = nlp(words[i+1])[0].pos
                if pos_1 in markov_pos:
                    if pos_2 in markov_pos[pos_1]:
                        markov_pos[pos_1][pos_2] += 1
                    else:
                        markov_pos[pos_1][pos_2] = 1
                else:
                    markov_pos[pos_1] = {pos_2 : 1}

            if words[i] in markov_chain:
                if words[i+1] in markov_chain[words[i]]:
                    markov_chain[words[i]][words[i+1]] += 1
                else:
                    markov_chain[words[i]][words[i+1]] = 1
            else:
                markov_chain[words[i]] = { words[i+1] : 1 }

# Code for levenshtein distance between two words from https://www.scaler.com/topics/levenshtein-distance-python/
def dist(A, B):
    N, M = len(A), len(B)
    # Create an array of size NxM
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i
    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j], # Insertion
                    dp[i][j-1], # Deletion
                    dp[i-1][j-1] # Replacement
                )

    return dp[N][M]

def find_similar(s: str) -> str:
    shortest_dist = 999999999
    shortest_word = ""
    for key in markov_chain.keys():
        if key != s:
            if dist(s,key.lower()) < shortest_dist:
                shortest_dist = dist(s,key)
                shortest_word = key
    return shortest_word

def normalize_probability(model, t = 1):
    for key in model.keys():
        sum_prob = sum(model[key].values())
        for m in model[key].keys():
            model[key][m] = (model[key][m] / sum_prob) * t
    return model

def markov_files(files: list[str]):
    for f in files:
        markov_file(f)
        if PRINT_PROGRESS: print(f"\nFile:{f} done!")

def mix_probability_pos():
    bar = progressbar.ProgressBar()
    for k in bar(markov_chain.keys()):
        words = markov_chain[k]
        k_pos = nlp(k)[0].pos
        if not k_pos in markov_pos:
            continue
        for w in words.keys():
            w_pos = nlp(w)[0].pos
            if not w_pos in markov_pos[k_pos]:
                markov_chain[k][w] = 0
                continue
            markov_chain[k][w] = markov_chain[k][w] * markov_pos[k_pos][w_pos]

def main():
    global markov_pos
    global markov_chain
    
    global TRAIN_POS
    global TRAIN_MARKOV
    global REMOVE_COMMA
    global REMOVE_QUOT
    global PRINT_PROGRESS

    if len(markov_chain) == 0:
        print("WARNING: Markov Chain model is empty")
    
    if len(markov_pos) == 0:
        print("WARNING: POS model is empty")

    usage_text = '%(prog)s.py [-h] [options] [--start_word word] [filenames...]' if NOT_PYINSTALLER else '%(prog)s.py [-h] [-n N] [--start_word word]'
    parser = argparse.ArgumentParser(
                    prog='plbs_gen', usage=usage_text,
                    description="""
                                This program generates polish sentence (either random or starting with provided word).\nUses mainly Markov Chain algorithm and NLP model to make it better (maybe)
                                If starting word cannot be found inside markov dictionary, then the most similar one is found and used.
                                """,
                    epilog="""
                    Dataset that the released tool used is from Allegro's summarization dataset.
                    It can be found on https://huggingface.co/datasets/allegro/summarization-polish-summaries-corpus/tree/main
                    Additionally the book of robinson-crusoe has been used from https://wolnelektury.pl/katalog/lektura/robinson-crusoe/
                    All rights to them and thank god that it exists. Of course you can use your own training, more info should be in the repository
                    """)
    
    parser.add_argument("-n", type=int, default=12, help="Number of words to generate, by default %(default)s")
    parser.add_argument("--start_word", metavar="WORD" ,type=str, help="With what word should generated sentence start. If no starting word is provided with --start_word flag, then random word is choosen from \"trained\" markov chain")
    parser.add_argument("--no_nlp", action='store_true',default=False, help="Use this flag to not use NLP in the generated sentence (NLP is what makes it so slow but might give better results)")

    if NOT_PYINSTALLER:
        parser.add_argument("--train_mark", action='store_true',default=False, help="Use this flag to train markov model on specified files")
        parser.add_argument("--train_pos", action='store_true',default=False, help="Use this flag to train nlp model (POS markov chain) on specified files")
        parser.add_argument("--print_prog", action='store_true',default=False, help="Print progress during training")
        parser.add_argument("--remove_quot", action='store_true',default=False, help="Remote quotations from text during training")
        parser.add_argument("--remove_comma", action='store_true', default=False, help="Remote comma(,) from text during training")
        parser.add_argument("files" ,nargs="*", help="Paths to files to train on")
    
    args = parser.parse_args()
    NO_NLP = args.no_nlp

    if TRAIN_POS or TRAIN_MARKOV:
        TRAIN_MARKOV = args.train_mark
        TRAIN_POS = args.train_pos
        PRINT_PROGRESS = args.print_prog
        REMOVE_COMMA = args.remove_comma
        REMOVE_QUOT = args.remove_quot

    if (TRAIN_MARKOV or TRAIN_POS) and len(args.files) == 0:
        print("ERROR: Specified training flag but not provided any files")
        return
    
    if TRAIN_MARKOV or TRAIN_POS:
        markov_files(args.files)

    if TRAIN_MARKOV: # TODO Add POS values to words in dictionary for a speed up if possible
        path = "src/_markov_model.p" if NOT_PYINSTALLER else os.path.join(sys._MEIPASS, "src/_markov_model.p")
        if len(markov_chain) > 0: pickle.dump( markov_chain, open(path, "wb") )
        if not TRAIN_POS: return
    
    if TRAIN_POS:
        path = "src/_pos_model.p" if NOT_PYINSTALLER else os.path.join(sys._MEIPASS, "src/_pos_model.p")
        if len(markov_chain) > 0: pickle.dump( markov_pos, open("src/_pos_model.p", "wb") )
        return

    key = ""
    if not args.start_word:
        key = random.choices(list(markov_chain.keys()),k=1)[0]
    else:
        key = args.start_word
    for i in range(args.n):
        print(key, end=" ")
        if not key in markov_chain:
            key = find_similar(key.lower())
            if not key in markov_chain: raise Exception("Not enough data to create chain. Can't find any similar word")
        
        probabilities = list(markov_chain[key].values())
        if not NO_NLP:
            words = list(markov_chain[key].keys())
            pos_probabilities = list(markov_pos[nlp(key)[0].pos])
            k_pos = nlp(key)[0].pos
            for i in range(len(probabilities)):
                w_pos = nlp(words[i])[0].pos
                if not w_pos in markov_pos[k_pos]:
                    probabilities[i] = probabilities[i] * 0
                    continue
                probabilities[i] = probabilities[i] * markov_pos[k_pos][w_pos]

        key = random.choices(list(markov_chain[key].keys()),weights=probabilities,k=1)[0]
    
    print("")

# No idea why would i need it here, i wont make a module out of it or anything but well why not
if __name__ == "__main__":
    main()
else:
    print("ERROR why not main?")