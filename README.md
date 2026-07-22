## Execution order

From the project root

```
python -m pip install -r requirements.txt
python -m scripts.inspect_ontology
python -m scripts.build_index
python -m scripts.test_ner
python -m scripts.test_candidates
python -m scripts.predict
python -m scripts.package
```