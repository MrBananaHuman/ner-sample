# Example code

## Requirements

torch==1.7.0
transformers==3.5.1
cython
kss==1.3.1

## How to use

### 1. Prepare data

```bash
.
├── ...
├── data
│   ├── librewiki.txt
│   ├── literature.txt
│   └── wiki.txt
└── ...
```

Data should be in one line for each context!!

<p float="left" align="left">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/101527542-7c392500-39d1-11eb-8fac-a08296f67556.png" />  
</p>

### 2. Run script

```bash
python3 run_ner.py --filename wiki.txt --device 0
```
