import csv
from collections import defaultdict
import itertools
import numpy as np


def read_csv(path):
    
    #assumes the order of the columns are - annotator, label, span_start, span_end and no header
    data = []

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([row[0],row[1],row[2],row[3]])
            
    return data

def nominal_metric(a, b):
    return a != b

def interval_metric(a, b):
    return (a-b)**2

def ratio_metric(a, b):
    return ((a-b)/(a+b))**2

def add_spans_with_no_label(spans,c_length):

    spans.sort(key=lambda x: x[1])
    result_spans = []
    previous_end = 0
    
    for label, start, end in spans:

        if start > previous_end:
            result_spans.append([0, previous_end, start])
        
        result_spans.append([label, start, end])
        previous_end = end
    
    #add a final segment
    if previous_end < c_length:
        result_spans.append([0, previous_end, c_length])
    
    return result_spans

def spans_to_index(data_dict):

    index_dict = {}

    for annotator, spans in data_dict.items():
        index_dict[annotator] = defaultdict()        
        span_id = 1
        for span in spans:
            index = range(span[1],span[2]+1)
            for i in index:
                index_dict[annotator][i] = (span_id,span[0],span[1],span[2])
            
            span_id += 1

    return index_dict

def data_to_dict(data_list,c_length):

    data_dict = defaultdict(list)

    #length for keeping track of index
    L = 0

    labels = [span[1] for span in data_list]
    label_to_id = {l:i+1 for i,l in enumerate(sorted(list(set(labels))))}
    id_to_label = {i:l for l,i in label_to_id.items()}

    #list of data -> data dict with [annotator]:[[label,span_start,span_stop],..]
    for annotated_span in data_list:
        annotator = annotated_span[0]
        label = label_to_id[annotated_span[1]]
        labels.append(label)
        span_start = int(annotated_span[2])
        span_stop = int(annotated_span[3])

        data_dict[annotator].append([label,span_start,span_stop])

        if span_stop > L:
            L = span_stop

    for annotator,spans in data_dict.items():
        data_dict[annotator] = add_spans_with_no_label(spans,c_length)
    
    return data_dict,id_to_label

def coincidence_matrices(data_dict,span_dict):

    #nr of annotators
    m = len(span_dict.keys())

    #the end of the last segment = the length of the continuum
    c_length = list(data_dict.values())[0][-1][-1]

    #pairable values
    l_all = m * c_length

    ###observed coincidences###
 
    obs_matrix = defaultdict(int)
    obs_margins = defaultdict(int)
    
    #equation 1, fig. 1

    for annotator_i, units_g in data_dict.items():
        for annotator_j,_ in data_dict.items():
            if annotator_i != annotator_j:
                for unit_g in units_g:
                    
                    #look up possible overlap in index_dict
                    possible_overlap = []

                    for i in range(unit_g[1],unit_g[2]):
                        possible_overlap.append(span_dict[annotator_j][i])
                    
                    for unit_h in list(set(possible_overlap)):

                        #intersection between unit g and h      
                        l_ck = max(0, min(unit_g[2],unit_h[3]) - max(unit_g[1],unit_h[2]))

                        if l_ck > 0:
                            c = unit_g[0]
                            k = unit_h[1]
                            obs_matrix[(c,k)]+= l_ck/(m-1)
   

    obs_margins = defaultdict(int)

    for k,v in obs_matrix.items():
        obs_margins[k[0]]+= v

    ###expected coincidences###   

    #equation 4 
    exp_matrix = defaultdict(float)

    #sum denominator equation 4
    sum_all_spans = 0
    #sum numerator equation 4
    sum_values = defaultdict(float)

    for spans in data_dict.values():
        for label,start,end in spans:
            sum_all_spans += (end - start) ** 2 if label != 0 else (end - start)
            sum_values[label] += (end - start) ** 2 if label != 0 else (end - start)
      
    all_label_combos = itertools.combinations_with_replacement(sum_values.keys(),2)

    #equation 4 for for all possible combinations of values c,k

    denominator = l_all**2 - sum_all_spans

    for c,k in all_label_combos:
        
        if c==k:
            e_ck = l_all * ((obs_margins[c]*obs_margins[k] - sum_values[c]) / denominator )
        else:
            e_ck = l_all * ((obs_margins[c] * obs_margins[k]) / denominator )
            
        exp_matrix[(c,k)] = e_ck
        exp_matrix[(k,c)] = e_ck
    
    exp_margins = defaultdict(float)
    
    for k,v in exp_matrix.items():
        exp_margins[k[0]] += v

    return obs_matrix,obs_margins,exp_matrix,exp_margins


def coincidence_matrices_wo_zero(obs_matrix,obs_margins,data_dict,index_dict):

    labels =  [k for k in obs_margins.keys() if k != 0]
    label_combos = list(itertools.combinations_with_replacement([k for k in labels if k!=0],2))
    
    #nr of annotators
    m = len(data_dict.keys())

    ###observed margins - without spans without annotation (label 0)

    cu_obs_margins = defaultdict(int)

    for value,margin in obs_margins.items():
        if value!=0:
            cu_obs_margins[value] = margin - obs_matrix[(value,0)]

    #pairable values
    l_all = sum(v for v in cu_obs_margins.values())

    ###expected coincidences - only intersections,not spans without annotation (label 0)

    #equation 10, part of denominator
    sum_all_spans = 0
    #equation 10, part of numerator
    sum_values = defaultdict(int)
    
    for annotator_i, units_g in data_dict.items():
        for annotator_j, units_h in data_dict.items():
            if annotator_i != annotator_j:
                for unit_g in units_g:
                    if unit_g[0] != 0:

                        #look up possible overlap in index_dict
                        possible_overlap = []

                        for i in range(unit_g[1],unit_g[2]+1):
                            unit_h = index_dict[annotator_j][i]
                            if unit_h[1] != 0:
                                possible_overlap.append(unit_h)
                        
                        intersect = 0
                        for unit_h in list(set(possible_overlap)):
                            intersect += max(0, min(unit_g[2],unit_h[3]) - max(unit_g[1],unit_h[2]))

                        intersect = intersect**2
                        sum_all_spans += intersect
                        sum_values[unit_g[0]] += intersect
    
    #equation 10, denominator
    denominator = l_all - (sum_all_spans/(l_all*(m-1)))

    cu_exp_matrix = {}

    #equation 10
    for c,k in label_combos:
        if c != k:
            e_ck = cu_obs_margins[c]*cu_obs_margins[k] / denominator
        
        else:
            e_ck = ((cu_obs_margins[c]*cu_obs_margins[k]) - (sum_values[c]/(m-1))) / denominator
    
        cu_exp_matrix[(c,k)] = e_ck
        cu_exp_matrix[(k,c)] = e_ck

    cu_exp_margins = defaultdict(float)
    
    for k,v in cu_exp_matrix.items():
        cu_exp_margins[k[0]] += v
    
    return cu_exp_matrix,cu_exp_margins,cu_obs_margins

def bar_u_alpha(obs_matrix,obs_margins,exp_matrix,exp_margins):

    """By adding a bar '|' to the subscript of ua, we are indicating that the |ua-coefﬁcient
    measures the reliability of a binary distinction between relevant and irrelevant matter,
    between the identiﬁed units taken together, and the gaps between units taken together. If
    the total amount of relevant matter informs the answer to a research question, |ua assesses
    the reliability of this amount."""
    
    if obs_margins[0] == 0:
        #print('No spans without annotation')
        b_u_alpha = 'NA'
    
    else:
        
        #disagreement zero-non_zero, fig.5
        do = obs_margins[0]-obs_matrix[0,0]
        de = exp_margins[0]-exp_matrix[0,0]
    
        #equation 6
        b_u_alpha = 1 - (do/de)

        print('bu_alpha: ',round(b_u_alpha,3))

    return b_u_alpha

def cu_alpha(obs_matrix,obs_margins,cu_obs_margins,cu_exp_matrix,labels,metric):

    """The cua-coefﬁcient focuses on what ua hides by its broadness and |ua ignores by summing
    all valued units into one category. Adding a ''c'' to ua's subscript is intended to indicate
    cua's focus on the coding of intersecting units, on their assigned values:"""

    labels =  [k for k in obs_margins.keys() if k != 0]

    #pairable values
    l_all = sum(v for v in cu_obs_margins.values())
    
    ### calclulate cu_alpha - equation 11

    if metric == nominal_metric:
        do = l_all - sum([obs_matrix[(c,c)] for c in labels])
        de = l_all - sum([cu_exp_matrix[(c,c)] for c in labels])
    else:
        do = sum([(metric(c,k))*obs_matrix[(c,k)] for c in labels for k in labels]) 
        de = sum([(metric(c,k))*cu_exp_matrix[(c,k)] for c in labels for k in labels]) 
    
    c_u_alpha = 1 - (do/de)
    if np.isclose(c_u_alpha,0):
        c_u_alpha = 0

    #coverage of the continuum
    coverage = 100 * l_all/sum(v for v in obs_margins.values())

    print('cu_alpha: ',round(c_u_alpha,3),' covering ',round(coverage,2),'% of the continuum')
            
    return c_u_alpha, coverage


def ku_alpha(obs_matrix,obs_margins,cu_obs_margins,cu_exp_matrix,cu_exp_margins,metric,id_to_label):

    """The last member of the ua-family of reliability coefﬁcients for unitized continua is the
    reliability of individual values k != 0, denoted by adding its name in parenthesis to ua. The
    (k)ua-coefﬁcient enables researchers to obtain a more detailed picture of the sources of
    unreliability associated with speciﬁc categories or values assigned to units."""

    labels = sorted(list(id_to_label.keys()))

    k_dict = {}

    for k in labels:

        #if there are no annotations for label k
        if cu_obs_margins[k] == 0:
            k_u_alpha = 'NA'

        #equation 13b
        elif metric == nominal_metric :
            e_k = cu_exp_margins[k]
            l_k = cu_obs_margins[k]
            l_kk = obs_matrix[(k,k)]
            e_kk = cu_exp_matrix[(k,k)]
            k_u_alpha = 1 - e_k/l_k * (l_k-l_kk) / (e_k-e_kk)
            
            if np.isclose(k_u_alpha,0):
                k_u_alpha = 0

        #equation 13a
        else:
            e_k = cu_exp_margins[k]
            o_sum_all = sum([(metric(c,k))*obs_matrix[(c,k)] for c in labels])
            l_k = cu_obs_margins[k]
            e_sum_all = sum([(metric(c,k))*cu_exp_matrix[(c,k)] for c in labels])

            do = e_k * o_sum_all  
            de = l_k * e_sum_all
            k_u_alpha = 1 - (do/de)

            if np.isclose(k_u_alpha,0):
                k_u_alpha = 0
        
        #coverage of the continuum
        coverage = 100*cu_obs_margins[k]/obs_margins[k] 
        k_dict[k] = (k_u_alpha,coverage)

        print('ku_alpha for label',id_to_label[k],' ',round(k_u_alpha,3),', covering ',round(coverage,2),'% of the spans labeled ',k)
    
    return k_dict

def u_alpha(data,c_length,metric_v):
    
    '''Calculates Krippendorff's unitized alpha for spans:
       the reliability or inter-annotator agreement 
       of partitioning and labeling a continuum.

       Based on unitized k-alpha as described in 'On the reliability 
       of unitizing textual continua: Further developments'
       https://link.springer.com/article/10.1007/s11135-015-0266-1

       Expects a list of of lists, where each list represents
       a span of the form : [Annotator,Label,Span_start,Span_end]
       Spans should not overlap within one annotator. 
       Assumes the segments between spans, spans without annotation,
       are not added. 
       Length of the continuum (all of it, with unannotated segments)
       should be provided.
       A metric for calculating the  
       
    
       Returns four versions of alpha:
       1. u-alpha   - Reliability for all spans and labels, considers 
                    both span overlap and label.
       2. b-u-alpha - Reliability of annotated spans vs. segments w.o. annotation.
                    Labels are not taken into consideration.
       3. c-u-alpha - Reliability of the intersections of annotated spans.
                    Also returns coverage, how much of the continuum is included.  
       4. k-u-alpha - Reliability of each label separately, considers 
                    only intersections of annotated spans. Also returns coverage, 
                    how much of the spans labeled with label k is included. 

    ---
       
    Args:
        data: a list of spans, [[Annotator,Label,Span start,Span end],..]
        length: the length of the continuum
        metric: name of the metric for calculating the difference between labels,can be 
                'nominal','interval' or 'ratio', default 'nominal'
    
    
    Returns
        u-alpha and b-u-alpha as float, (c-u-alpha,coverage), k_dict: a dict of the form {value:(alpha,coverage)}
    
    '''
    
    # u_alpha for all identified segments, nominal
    """The ua-coefﬁcient applies to all segments of a unitized continuum, which includes not only
    the identiﬁed units, but also the gaps left between units. Because gaps have no identiﬁed
    structure and the distinction between c = no annot and c != no annot implies no ordering, ua is limited
    to the use of the nominal metric deﬁned by"""

    #data = a list of lists, with each list representing an annotated span: [Annotator,Label,Span start,Span end]

    data_dict,id_to_label = data_to_dict(data,c_length)

    #index_dict: annotator:[index:spans occuring at that index,..],..
    index_dict = spans_to_index(data_dict)

    metric_dict = {'nominal':nominal_metric,'interval':interval_metric,'ratio':ratio_metric}

    metric = metric_dict.get(metric_v,nominal_metric)

    ###observed  & expected coincidences###

    obs_matrix,obs_margins,exp_matrix,exp_margins = coincidence_matrices(data_dict,index_dict)
    cu_exp_matrix,cu_exp_margins,cu_obs_margins = coincidence_matrices_wo_zero(obs_matrix,obs_margins,data_dict,index_dict)

    ###alphas###

    #u-alpha# section 5.1  

    #pairable values
    l_all = sum(v for v in obs_margins.values())

    #equation 5a
    #sum all intersections - the diagonals of the matrices (agreement) 

    do = l_all - sum(v for k,v in obs_matrix.items() if k[0] == k[1])
    de = l_all - sum(v for k,v in exp_matrix.items() if k[0] == k[1])

    u_alpha = 1 - do/de

    print('u alpha: ',round(u_alpha,3))

    #b_u_alpha - irrelevant segments vs identified segments, section 5.2
      
    b_u_alpha = bar_u_alpha(obs_matrix,obs_margins,exp_matrix,exp_margins)

    labels =  [k for k in obs_margins.keys() if k != 0]

    #if there is only one valued label, c_u and k_u are not applicable
    if len(labels) == 1:
        c_u_alpha = 'NA'
        coverage = 0 
        k_dict = {}
        print('c_u_alpha and k_u_alpha are not applicable for one label')
    
    else:
        #c_u_alpha - intersections of valued units section 5.3
        
        print('metric for cu and ku alpha:',metric.__name__)
        
        c_u_alpha, coverage = cu_alpha(obs_matrix,obs_margins,cu_obs_margins,cu_exp_matrix,data_dict,metric)

        #(k)_u_alpha - alpha of intersections of each value

        k_dict = ku_alpha(obs_matrix,obs_margins,cu_obs_margins,cu_exp_matrix,cu_exp_margins,metric,id_to_label)

    return b_u_alpha,u_alpha, (c_u_alpha,coverage), k_dict


if __name__ == "__main__":
    
    """u-alpha—u for unitizing—is applicable to partitions of continua into mutually exclusive segments of different kinds. 
    |-ualpha reveals the reliability of distinguishing relevant segments from irrelevant ones. cu-alpha is designed to establish 
    the reliability of coding segments of unequal lengths but ignoring the potential unreliabilities of unitizing, and 
    (k)u-alpha to evaluate the reliabilities of individual categories or values"""


    #annotated_spans from csv csv
    ##assumes the order of the columns are - annotator, label, span_start, span_end and no header
    #path = 'something'
    #annotated_spans = read_csv(path)

    annotated_spans = [['Alex', '1', '2', '5'],['Alex', '1', '9', '11'],['Alex', '2', '12', '20'],['Paul', '1', '2', '5'],['Paul', '2', '7', '15'],['Susan', '1', '2', '5'],['Susan', '2', '10', '15']]

    #length of the continuum
    c_length = 20

    alphas = u_alpha(annotated_spans,c_length,nominal_metric)

    #print(alphas)