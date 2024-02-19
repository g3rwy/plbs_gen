# Polish bullshit generator
This is a small program to generate sentences in polish, they do not make any sense and they do not answer a question or talk with you like ChatGPT. It is based on markov chain so it doesnt really understand what it says but sometimes the sentences make sense.
## Usage (--help or -h)
```
usage: plbs_gen.py [-h] [options] [--start_word word] [filenames...]

This program generates polish sentence (either random or starting with provided word). Uses mainly Markov Chain algorithm and NLP model to make it better (maybe) If starting word cannot be found inside markov
dictionary, then the most similar one is found and used.

positional arguments:
  files              Paths to files to train on

options:
  -h, --help         show this help message and exit
  -n N               Number of words to generate, by default 12
  --start_word WORD  With what word should generated sentence start. If no starting word is provided with --start_word flag, then random word is choosen from "trained" markov chain
  --no_nlp           Use this flag to not use NLP in the generated sentence (NLP is what makes it so slow but might give better results)
  --train_mark       Use this flag to train markov model on specified files
  --train_pos        Use this flag to train nlp model (POS markov chain) on specified files
  --print_prog       Print progress during training
  --remove_quot      Remote quotations from text during training
  --remove_comma     Remote comma(,) from text during training

Dataset that the released tool used is from Allegro's summarization dataset. It can be found on https://huggingface.co/datasets/allegro/summarization-polish-summaries-corpus/tree/main Additionally the book of
robinson-crusoe has been used from https://wolnelektury.pl/katalog/lektura/robinson-crusoe/ All rights to them and thank god that it exists. Of course you can use your own training, more info should be in the
repository
```

## Installation and Setup
You should be able to use it out of the box bundled up (using pyinstaller), the zip is in the [release section](https://github.com/g3rwy/plbs_gen/releases)

But if you want to install it your way i would:
- Clone repository : `git clone https://github.com/g3rwy/plbs_gen` and enter directory `cd plbs_gen`
- Create virtual environment inside `python -m venv`
- Activate the enviroment `source bin/your_shell.sh` (use the appropriate script for your shell)
- Install the requirements `pip install -r requirements.txt`
- Get more info with `python src/__main__ -h`
- Example of generating sentence with 24 words starting with Dom: `python src/__main__ -n 24 --start_word Dom`
You can also bundle up all that into into an executable and folder with data with pyinstaller, for this i provided `bundle.sh` script to do it automatically, the return bundle will appear in `dist/plbs_gen`

## Data used
Like its said in the help message, i used allegro's dataset for summarization, its pretty big so the marko chain should know enough words.
