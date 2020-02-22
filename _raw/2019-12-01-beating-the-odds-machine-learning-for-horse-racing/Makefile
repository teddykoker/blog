

all: raw.md
	pandoc raw.md -t markdown-citations -s --atx-headers \
			--bibliography bib.bib --csl footnotes.csl -o index.md
