'''
BMI 550: Applied BioNLP
Assignment 1: Rule-based system (Part 3)

SUMMARY:
This script accepts a single .xlsx file, specifically the "TEXT" column, and generates:
--> For each row, [CUI +/- negation] predicted labels for each phrase from within the "TEXT" column.


OUTPUT FORMAT: 
the system will output a text file that contains, tab separated, the id of a post, 
    the symptom expression (or entire negated symptom expression), the CUI for the symptom, 
    and a flag indicating negation (i.e., 1 if the symptom expression is a negated one, 0 otherwise).

A NOTE ON REPEATED CUIs IN A POST: 
" Systemevaluationwillonlyconsiderifapost(i)containsaspecificsymptom(i.e., 
the evaluation script will look for CUIs or 'Other'/'C0000000'), and (ii) if the negation flag on the CUI. 
Please see Week 4, Lecture 2 (Thursday) and the IAA_Calculator.py to see how the negation flag can be appended to a CUI.


@author Chase Fensore
email: abeed.sarker@dbmi.emory.edu

'''

import pandas as pd
from collections import defaultdict
import re
import nltk
# from nltk.tokenize import sent_tokenize

# For fuzzy matching
import Levenshtein
from fuzzywuzzy import fuzz
import thefuzz
import itertools

import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def load_supervision_data(f_path_list, preprocess):
    """
    Supervision data from annotated files of Reddit posts.
    File format must exactly follow that shown in files s1.xlsx - s15.xlsx.

    :param
    :returns Dictionary of: key: CUI, value: python list of Symptom Expressions. 
        We don't pay attention to the 'Negation Flag' column here. All CUI <-> Symptom Expression pairs are used.

    """
    rules_dict = defaultdict(list)
    for f_path in f_path_list:
        sup_df = pd.read_excel("./data/annots/" + f_path)
        for index,row in sup_df.iterrows():
            id_ = row['ID']
            if not pd.isna(row['Symptom CUIs']) and not pd.isna(row['Symptom Expressions']):
                cuis = row['Symptom CUIs'].split('$$$')[1:-1]
                symp_exprs = row['Symptom Expressions'].split('$$$')[1:-1]
                for cui,symp_exprs in zip(cuis,symp_exprs):
                    # NOTE: Before adding new symptom expression, ENFORCE 1:N relationship (CUI:symptom expression). We only keep 1st match.
                    #print([se[0] for se in rules_dict.values()])

                    # NOTE: try not enforcing 1:N relationship. What happens? Second, what if we keep repeated Ses & avg top 2 sim w/in a CUI?
                    all_vals = set([se[0] for se in rules_dict.values()])
                    if(symp_exprs.lower() not in all_vals):
                        res = str(symp_exprs).lower() # Lowercase (always)
                        if preprocess == 'pp1': # PP1: remove punctuation
                            res = res.translate(str.maketrans("", "", string.punctuation)) # Punctuation removal
                            res = ' '.join((nltk.word_tokenize(res)))
                        elif preprocess == 'pp2': # PP2: PP1 & Word Tokenize & Stem words (Porter stemmer)
                            stemmer = PorterStemmer()
                            res = list(nltk.word_tokenize(res))
                            res = [stemmer.stem(token) for token in res]
                            res = ' '.join(res) # To keep res a string.
                        elif preprocess == 'none': # Only lowercasing then word tokenize.
                            res = ' '.join((nltk.word_tokenize(res)))
                            
                        # Apply any other supervision data preprocessing here. Should be same as test data preprocessing.
                        #res = str(nltk.word_tokenize(res))
                        rules_dict[cui].append(res) # lower-case symptom expression.

    return rules_dict



def load_test_data(test_f_path):
    """
    Test data from UN-ANNOTATED files of Reddit posts.
    Input file format must exactly follow that shown in files s1.xlsx - s15.xlsx, except gold labels may not be present.
     ***PROCEDURE FOR PREPROCESSING***:
        - do not stem words
        - do not remove stop words
        - DO lowercase everything (bc we don't want to match w non-lowercased words)
        - since lexicon has punc & symbols, keep punc & symbols BUT REPLACED ampersands etc??? (check on this in lexicon)

    :param test_f_path, the path to the input test file.
    :returns dataframe of un-annotated columns: 
        ID (String)	TEXT (String that's been lowercased)
    """
    test_df = pd.read_excel(test_f_path)
    # Lowercase TEXT column
    test_df['TEXT'] = test_df['TEXT'].str.lower() 

    #dict_test = defaultdict(list)
    #for index,row in test_df.iterrows():
    #    if not pd.isna(row['ID']) and not pd.isna(row['TEXT']):
    #        id_ = row['ID']
    #        text = row['TEXT']
    #        for id_,text in zip(id_,text):
    #            dict_test[id_].append(str(text).lower()) # lower-case "TEXT" column
    #            # Apply any other supervision data preprocessing here. Should be same as test data preprocessing.  
    return test_df




def run_sliding_window_through_text(words, window_size):
    """
    NOTE: Borrowed from windowing_and_thresholding.py
    Generate a window sliding through a sequence of words
    :param words:
    :param window_size:
    """
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

# index, test_df, rules_dict, _min_pred_thresh_
def match_dict_similarity(index, test_df, rules_dict, _min_pred_thresh_, sim_metric, preprocess):
    '''
    NOTE: Borrowed from windowing_and_thresholding.py
    This function performs prediction of 'Standard Symptom' and 'Standard CUIs' cell for ONE ROW ONLY.
    :param index: row index within test_df to perform predictions on 'TEXT' cell of.
    :param expressions: i.e. variants of symptom expressions.
    :param test_df: 
    :param: _min_pred_thresh_
    :param: sim_metric
    :return: a 4-tuple containing the predicted columns to be output to results.xlsx.
    '''
    threshold = _min_pred_thresh_
    #max_similarity_obtained = -1
    
    text = test_df.at[index, 'TEXT']
    #print("\n\n\nTEXT: \n", text)

    # Preprocessing options (on TEST data)
    if preprocess == 'pp1': # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokenized_text = list(nltk.word_tokenize(text))
    # ALWAYS Word tokenize
    #text = nltk.word_tokenize(text)
    if preprocess == 'pp2':
        tokenized_text = list(nltk.word_tokenize(text))
        stemmer = PorterStemmer()
        #text = str(nltk.word_tokenize(text))
        tokenized_text = [stemmer.stem(token) for token in tokenized_text]
        #tokenized_text = ' '.join(tokenized_text) # To keep res a string.
    elif preprocess == 'none':
        tokenized_text = list(nltk.word_tokenize(text)) # Only lowercase then word tokenize.
        

    #print("\n\n\nTOK TEXT\n: ", tokenized_text)
    SS_output_list = [] #  list(standard symptom)
    SE_output_list = [] # list: Holds fuzzy matches actually present in 'TEXT'
    CUI_output_list = [] # list(CUI)
    CUI_negation_list = []

    # -- only accept the current CUI if 1 or more of its Standard Symptom variants has sim score > threshold. 
    # expressions only gives 'Standard Symptoms' list for the current CUI.
    for cui, expressions in rules_dict.items():
        # Go through each expression
        best_exp = '' # Standard Symptom w highest sim w/in current CUI. 
        best_match = '' # (not needed) Symptom Expression *actually* present in TEXT.
        best_exp_sim = -1
        
        for exp in expressions:
            #create the window size equal to the number of word in the expression in the lexicon
            exp_max_sim_across_windows = -1
            size_of_window = len(exp.split()) 
            for window in run_sliding_window_through_text(tokenized_text, size_of_window):
                window_string = ' '.join(window)

                similarity_score = -1
                if sim_metric == 'Levenshtein':
                    similarity_score = Levenshtein.ratio(window_string, exp) # Metric 1: Levenshtein ratio
                elif sim_metric == 'token_sort_ratio':
                    similarity_score = fuzz.token_sort_ratio(window_string, exp) # Metric 2: Token sort ratio
                elif sim_metric == 'simple_ratio':
                    similarity_score = fuzz.ratio(window_string, exp) # Metric 3: Simple ratio


                #print("SIM SCORE: ", similarity_score)
                if similarity_score >= threshold:
                    #print (similarity_score,'\t', exp,'\t', window_string)
                    if similarity_score>exp_max_sim_across_windows:
                        exp_max_sim_across_windows = similarity_score
                        best_match = window_string # For SE_output_list
                        best_exp_sim = similarity_score 
                        best_exp = exp # For SS_output_list
                        print("\n\nSIM: ", exp_max_sim_across_windows)
        # For current CUI, record the CUI & its best exp (ONLY if best_exp_sim > -1, implying that best_exp_sim > threshold)
        if(best_exp_sim >= threshold):
            SS_output_list.append(best_exp)
            SE_output_list.append(best_match)
            CUI_output_list.append(cui)


        #############
        # Get tuples of info on locations of SE_output_list in 'TEXT':
        se_tuples = [] # list(tuple). tuple: []
        for se in SE_output_list:
            #se_match_info = re.finditer(r'\b'+se+r'\b',text)
            start_idx = text.find(se)
            end_idx =  start_idx + len(se) - 1
            match_tuple = (text, se, se, start_idx, end_idx)
            # Only takes 1st match??
            se_tuples.append(match_tuple)

        # Now, check negations using: text, SS_output_list && CUI_output_list:
        CUI_negation_list = [] # Only stores 0/1, not the CUI.
        negations = load_negations()

        #("SS output: \n\n", SS_output_list)
        #print("SE OUTPUT:\n\n", SE_output_list)
        #print("CUIs:\n\n", CUI_output_list)

        #print("SE TUPLES:\n\n", se_tuples)
        for se, cui in zip(se_tuples, CUI_output_list):
            is_negated = False
            #text = se[0]
            cui = se[1] # Or just cui.
            expression = se[2]
            start = se[3]
            end = se[4]
            for neg in negations:

                for match in re.finditer(r'\b'+neg+r'\b', re.escape(text)):
                    #if there is a negation in the sentence, we need to check
                    #if the symptom expression also falls under this expression
                    #it's perhaps best to pass this functionality to a function.
                    # See the in_scope() function
                    is_negated = is_negated or in_scope(match.end(),text,expression)
            if is_negated:
                CUI_negation_list.append("1")
            if not is_negated:
                CUI_negation_list.append("0")     
        # Save exp in 'Standard Symptom' column, CUI in 'Standard CUIs' column. Just for current post ID.
        #print (best_match,exp_max_sim_across_windows)
        #0: Symptoms, 1: CUIs.
    return (SS_output_list, CUI_output_list, CUI_negation_list, SE_output_list)


def predict_test_CUIs(rules_dict, test_df, _min_pred_thresh_, sim_metric, preprocess):
    """
    Given the 'rules_dict', where key=CUI: function predicts CUIs within each row of "TEXT" in 'test_data.'
    NOTE: Negation is done in a separate function, & negations are stored in the "Negation Flag" column. 

    :params rules_dict:
             test_df: Pandas df with columns: ID || TEXT
             _min_pred_thresh_: Must be between 0-100. Used as the minimum fuzzy match score before predicting 'Other'/C0000000 CUI.
             sim_metric (String): similarity metric among the following (Levenshtein, token_sort_ratio, simple_ratio)
    :returns dataframe with columns:
            ID (String)	|| TEXT (String that's been lowercased) || *Symptom Expressions (extracted) || *Symptom CUIs (predicted)
    """
     
    # For each post (i.e. each row)...
    for index,row in test_df.iterrows():
        if not pd.isna(row['ID']) and not pd.isna(row['TEXT']):
            id_ = row['ID']
            # Check for (fuzzy) matches in test_df['TEXT'] using rules_dict.values(). Then back-calculate key of rules_dict.
            preds_tuple = match_dict_similarity(index, test_df, rules_dict, _min_pred_thresh_, sim_metric, preprocess)

            test_df.at[index, 'Standard Symptom'] = ''.join(['$$$' + item + '$$$' for item in preds_tuple[0]])
            test_df.at[index, 'Symptom CUIs'] = ''.join(['$$$' + item + '$$$' for item in preds_tuple[1]])
            test_df.at[index, 'Negation Flag'] = ''.join(['$$$' + item + '$$$' for item in preds_tuple[2]])  # CUI_negation_list
            test_df.at[index, 'Symptom Expressions'] = ''.join(['$$$' + item + '$$$' for item in preds_tuple[3]]) # SE_output_list
            #test_df.at[id_, 'Standard Symptom'] = preds_tuple[0] # test_df[id_, 'Standard Symptom'] = ['$$$' + item + '$$$' for item in preds_tuple[0]]
            #test_df.at[id_, 'Symptom CUIs'] = preds_tuple[1] # test_df[id_, 'Symptom CUIs'] = ['$$$' + item + '$$$' for item in preds_tuple[1]]
        else:
            break
    # Now 'test_df' has preds in columns: 'Standard Symptom' || 'Symptom CUIs'
    return test_df


def in_scope(neg_end, text, symptom_expression):
    '''
    NOTE: Borrowed from hw4 solution.

    Function to check if a symptom occurs within the scope of a negation based on some
    pre-defined rules.
    :param neg_end: the end index of the negation expression
    :param text:
    :param symptom_expression:
    :return:
    '''
    negations = load_negations()
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))
    # this is the maximum scope of the negation, unless there is a '.' or another negation
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])
    #Note: in the above we have to make sure that the text actually contains 3 words after the negation
    #that's why we are using the min function -- it will be the minimum or 3 or whatever number of terms are occurring after
    #the negation. Uncomment the print function to see these texts.
    #print (three_terms_following_negation)
    match_object = re.search(re.escape(symptom_expression), three_terms_following_negation)
    if match_object:
        period_check = re.search('\.', three_terms_following_negation)
        next_negation = 1000 #starting with a very large number
        #searching for more negations that may be occurring
        for neg in negations:
            # a little simplified search..
            if re.search(neg, text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            #if the period occurs after the symptom expression
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated




def load_negations():
    """
    NOTE: Borrowed from hw4 solution.

    Function to load negations.
    @returns 'negations,' a Python list of negation trigger words.
    """
    #loading the negation expressions
    negations = []
    infile = open('./data/neg_trigs.txt')
    for line in infile:
        negations.append(str.strip(line))
    return negations





def main():
    #  Please see Week 4, Lecture 2 (Thursday) and the IAA_Calculator.py to see how the negation flag can be appended to a CUI.
    #  'Other'/'C0000000' is not used. One limitation of my system is that the 'Other'/'C0000000' was not used during the annotation process.
    #   In other words, none of the annotation files used for training included 'Other'/'C0000000' as symptom labels, and instead always mapped to a CUI in 
    # ... annotated s-X.xlsx files.
    """
    ***RULE-BASED PROCEDURE***:
    
    **Inputs**:
    - We have supervision file (A): 'Symptom Expression' and 'CUI' columns from a file of labeled supervision data.
    - Negation prefixes file (./data/neg_trigs.txt).
    - List of all CUIs possible (cuilist.txt)
    
   **Outputs**: 
    - ex for 1 post: Given a row in .xlsx file, output a file './data/result.xlsx' with columns: 
        ID	TEXT	*Symptom CUIs	*Negation Flag [Where * indicates system predictions.]

    **To accomplish this**...
    1. Read in supervision data (from my post-consensus s2-resolved.xlsx file, or all s-X.xlsx files).
        - we also use ALL annotated files (s1 - s15), and compare results to s2-resolved.xlsx.

    2. [various methods tried] Using supervision data to "learn" a dictionary of rules for CUI mappings to Symptom Expression synonyms (1:N):
    - key: CUI(*ultimately the label being output) -> value: Python list of *synonyms* for the Symptom Expression.
    - NOTE: Theoretically, there can be an N:N relationship between CUI:Symptom Expression. 
        This can happen if the same Symptom Expression maps to a different CUI based on (a) annotator disagreement on CUI label or (b) different context.
        TODO: Here, we assume each Symptom Expression only belonds to 1 CUI. This assumption is enforced for this set of supervision data.


    **NEGATION PROCEDURE**: 
    1. Using load_negations() prefixes, write 0/1 whether each symptom expression identified is 0(not negated) or 1(negated).


    """
    # Supplementary CUI dictionary: COVID-Twitter-Symptom-Lexicon
    # Column 2: CUIs (no header, are repeated). Repeated for each symptom expression.
    # Column 3: Symptom expression.
    """
    symptom_dict = {}
    infile = open('./data/COVID-Twitter-Symptom-Lexicon.txt')
    for line in infile:
        items = line.split('\t')

        symptom_dict[str.strip(items[-1].lower())] = str.strip(items[1])
    print('Printing the contents of the dictionary')
    for k,v in symptom_dict.items():
        print (k,'\t',v)
    """
    

    ###################
    # First, read in supervision data. 
    # During this, we learn "rules" for CUI prediction in the form of a dictionary.
    f_path_list = ["s1.xlsx", "s2-resolved.xlsx", "s3.xlsx", "s4.xlsx",
                    "s5.xlsx", "s6.xlsx", "s7.xlsx", "s8.xlsx", "s9.xlsx", "s10.xlsx", "s11.xlsx", "s12.xlsx", "s13.xlsx", "s14.xlsx", "s15.xlsx"]
    rules_dict = load_supervision_data(f_path_list, preprocess='none') # pp1, pp2

    # Now, annotate test data with rule-based systems.
    test_df = load_test_data("./data/Assignment1GoldStandardSet.xlsx") # Assignment1GoldStandardSet.xlsx
    # Drop labels columns so we can predict them.
    test_df.drop('Symptom CUIs', axis=1, inplace=True)
    test_df.drop('Negation Flag', axis=1, inplace=True)
    # PERFORM TESTING of 'rules_dict' (Rule-based system) on 'test_df': 
    # Hyperparamters:
    # 1.) sim_metric options: (Levenshtein, token_sort_ratio, simple_ratio)
    # 2.) _min_pred_thresh_:
    #   0-1.0: Levenshtein
    #   0-100: token_sort_ratio, simple_ratio
    # 3.) preprocess: 
    #   none: lowercase then word tokenize
    #   pp1: lowercase, word tokenize, punctuation removal.
    #   pp2: lowercase, word tokenize, punctuation removal, stemming with a Porter stemmer.
    preds_df = predict_test_CUIs(rules_dict, test_df, _min_pred_thresh_= 90, sim_metric="token_sort_ratio", preprocess='pp2') # Levenshtein: 0 - 1.0 || Token sort ratio: 0-100
    preds_df.to_excel("./data/result.xlsx") # result.xlsx
    ###################


    # TODO: 
    # 1. run on Assignment1GoldStandardSet.xlsx. Run EvaluationScript on result & report results as "validaiton".
    # 2. Run on UnlabeledSet.xlsx. No labels, just submit with submission.


if __name__ == '__main__':
    main()

