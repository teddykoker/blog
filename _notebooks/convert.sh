#!/bin/bash

convert () {
    jupyter nbconvert $1  --config jekyll.py
}

convert $1
