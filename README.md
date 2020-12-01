# NER Sample Code

- `KoELECTRA-Base-v3`로 학습
- `창원대 NER 데이터셋 이용`
- `어절` 단위 태그 예측

## Prerequisite

`transformers` 버전은 반드시 지켜주세요

- torch==1.7.0
- transformers==3.5.1

## How to Use

- `test_naver_ner.py` 참고
- `str`, `List[str]` 모두 가능
- device 지정 가능 (cpu: -1)
- `ignore_labels=[]`로 세팅하면 `O` 라벨이 달린 것도 모두 볼 수 있음

```python
from transformers import ElectraTokenizer, ElectraForTokenClassification
from ner_pipeline import NerPipeline
from pprint import pprint

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-naver-ner")
model = ElectraForTokenClassification.from_pretrained("monologg/koelectra-base-v3-naver-ner")

ner = NerPipeline(model=model, tokenizer=tokenizer, ignore_labels=["O"], ignore_special_tokens=True, device=-1)

text = "문재인 대통령은 28일 서울 코엑스에서 열린 ‘데뷰 (Deview) 2019’ 행사에 참석해 젊은 개발자들을 격려하면서 우리 정부의 인공지능 기본구상을 내놓았다. 출처 : 미디어오늘 (http://www.mediatoday.co.kr)"
pprint(ner(text))

# Out
[{'entity': 'DAT-B', 'score': 0.9996743202209473, 'word': '2009년'},
 {'entity': 'DAT-I', 'score': 0.9999437928199768, 'word': '7월'},
 {'entity': 'ORG-B', 'score': 0.9999828934669495, 'word': 'FC서울을'},
 {'entity': 'LOC-B', 'score': 0.99980229139328, 'word': '잉글랜드'},
 {'entity': 'ORG-B', 'score': 0.9999563097953796, 'word': '프리미어리그'},
 {'entity': 'ORG-B', 'score': 0.999943196773529, 'word': '볼턴'},
 {'entity': 'ORG-I', 'score': 0.9996771812438965, 'word': '원더러스로'},
 {'entity': 'PER-B', 'score': 0.9999755620956421, 'word': '이청용은'},
 {'entity': 'ORG-B', 'score': 0.9999575614929199, 'word': '크리스탈'},
 {'entity': 'ORG-I', 'score': 0.9998330473899841, 'word': '팰리스와'},
 {'entity': 'LOC-B', 'score': 0.9999109506607056, 'word': '독일'},
 {'entity': 'ORG-B', 'score': 0.9999876022338867, 'word': '분데스리가2'},
 {'entity': 'ORG-B', 'score': 0.9999392628669739, 'word': 'VfL'},
 {'entity': 'ORG-I', 'score': 0.8798553943634033, 'word': '보훔을'},
 {'entity': 'DAT-B', 'score': 0.9999814033508301, 'word': '지난'},
 {'entity': 'DAT-I', 'score': 0.9999607801437378, 'word': '3월'},
 {'entity': 'ORG-B', 'score': 0.9999819993972778, 'word': 'K리그로'},
 {'entity': 'LOC-B', 'score': 0.8361089825630188, 'word': '서울이'},
 {'entity': 'ORG-B', 'score': 0.9971734881401062, 'word': '울산이었다'}]
```
