#!/bin/bash

convert () {
    # jupyter nbconvert $1  --config jekyll.py
    pandoc $1 -t markdown-citations -s --atx-headers \
        --bibliography bib.bib --csl footnotes.csl -o ../_drafts/$1
}

convert $1

