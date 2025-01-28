# Krippendorff's unitized alpha
Python implementation of Krippendorff's unitized alpha.

This implementation calculates Krippendorff's unitized alpha for spans -
the reliability or inter-annotator agreement of partitioning and labeling a continuum.
For example, calculating the agreement in a span labelling task where the contiuum could be a text. 

Based on unitized k-alpha as described in 'On the reliability of unitizing textual 
continua: Further developments'
https://link.springer.com/article/10.1007/s11135-015-0266-1

Example:

```python
#data: a list of annotated spans, [[Annotator,Label,Span start,Span end],..]
annotated_spans = [['Alex', '1', '2', '5'],['Paul', '1', '2', '5'],['Paul', '2', '10', '20'],['Susan', '2', '2', '5'],['Susan', '2', '15', '20']]


#length of the continuum/text
length = 60

#metric for calcultaing the difference between labels
metric = 'nominal'

u_alpha(anntotated_spans, length, metric)

#output
#u alpha:  0.474
#bu_alpha:  0.433
#metric for cu and ku alpha: nominal_metric
#cu_alpha:  0.779  covering  38.33 % of the continuum
#ku_alpha for label 1   0.769 , covering  95.45 % of the spans labeled  1
#ku_alpha for label 2   0.787 , covering  59.52 % of the spans labeled  2
```

In the example above, 
- u_alpha is the general agreement
- bu_alpha is the agreement between annotated spans and unannotated segments, disregarding the label
- cu_alpha is the agreement for labels, looking only at the overlapping parts of spans. (38.3% of the text/continuum is annotated by all annotators)
- ku_alpha is the agreement per label, looking only at overlapping segments of spans. (For label 1, 95.4% of the spans labeled 1 overlap)

Note that spans cannot overlap and that segments without annotation should not be included. 



