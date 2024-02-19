#!/bin/sh
python_ver="$(ls lib)"
pyinstaller --clean -y -n plbs_gen src/__main__.py --add-data="src/_markov_model.p:src" --add-data="src/_pos_model.p:src" --add-data="lib/${python_ver}site-packages/pl_core_news_sm:pl_core_news_sm" 