import numpy as np
import sys
import argparse
import os
import json
import re
import string
import csv

def extract1( comment, slang_words, bristol, warringer, alt_id, center_id, right_id, left_id, alt_data, center_data, right_data, left_data ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats_local = np.zeros(173+1)

    #1 Count first person pronouns  
    tokens_1 = ['I', 'me', 'mine', 'we', 'us', 'our', 'ours'] 
    temp_1 = ''

    for i in tokens_1:
        regexp_1 = r' '+i+'(?=[/])'
        temp_1 = re.findall(regexp_1, comment['body'])
        feats_local[1-1]+=len(temp_1)
        temp_1 = list() 

    print('No. of first person pronouns: ' + str(feats_local[1-1]))

    #2 Count second person pronouns  
    tokens_2 = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    temp_2 = ''

    for i in tokens_2:
        regexp_2 = r' '+i+'(?=[/])'
        temp_2 = re.findall(regexp_2, comment['body'])
        feats_local[2-1]+=len(temp_2)
        temp_2 = list() 

    print('No. of second person pronouns: ' + str(feats_local[2-1]))

    #3 Count third person pronouns  
    tokens_3 = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
    temp_3 = ''

    for i in tokens_3:
        regexp_3 = r' '+i+'(?=[/])'
        temp_3 = re.findall(regexp_3, comment['body'])
        feats_local[3-1]+=len(temp_3) 
        temp_3 = list() 

    print('No. of third person pronouns: ' + str(feats_local[3-1]))

    #4 Count coordinating conjunctions
    temp_4 = ''

    regexp_4 = r'(?<=[/])CC'
    temp_4 = re.findall(regexp_4, comment['body'])
    feats_local[4-1]+=len(temp_4) 

    print('No. coordinating conjunctions: ' + str(feats_local[4-1]))

    #5 Count past tense tokens 
    temp_5 = ''

    regexp_5 = r'(?<=[/])VBD'
    temp_5 = re.findall(regexp_5, comment['body'])
    feats_local[5-1]+=len(temp_5) 

    print('No. of past tense tokens: ' + str(feats_local[5-1]))

    #6 Count future tense tokens 
    tokens_6 = ['\'ll', 'will', 'gonna'] 

    regexp_6a = r'\S+'+tokens_6[0]+'(?=[/])'
    temp_6a = re.findall(regexp_6a, comment['body'])
    feats_local[6-1]+=len(temp_6a)

    regexp_6b = r' '+tokens_6[1]+'(?=[/])'
    temp_6b = re.findall(regexp_6b, comment['body'])
    feats_local[6-1]+=len(temp_6b)

    regexp_6c = r' '+tokens_6[2]+'(?=[/])'
    temp_6c = re.findall(regexp_6c, comment['body'])
    feats_local[6-1]+=len(temp_6c)

    print('No. of future tense tokens: ' + str(feats_local[6-1]))

    #7 Count the commas
    temp_7 = ''

    # Before /
    regexp_7 = r',(?=[/])'
    temp_7 = re.findall(regexp_7, comment['body'])
    feats_local[7-1]+=len(temp_7)
    temp_7 = list()
    
    # After /
    regexp_7 = r'(?<=[/]),'
    temp_7 = re.findall(regexp_7, comment['body'])
    feats_local[7-1]+=len(temp_7) 

    print('No. of commas: ' + str(feats_local[7-1]))

    #8 Count multiple character tokens
    temp_8 = ''

    for i in string.punctuation:
        if (i !=  '(' and i !=  ')' and i !=  '^' and i !=  '\\' and i !=  '/'):
            regexp_8 = r'['+i+']['+i+'](?=[/])'
            temp_8 = re.findall(regexp_8, comment['body'])
            feats_local[8-1]+=len(temp_8) 
            temp_8 = list()

    print('No. of multiple character tokens: ' + str(feats_local[8-1]))

    #9 Count common noun tokens
    temp_9a = list()
    temp_9b = list()

    regexp_9a = r'(?<=[/])NN'
    temp_9a = re.findall(regexp_9a, comment['body'])
    feats_local[9-1]+=len(temp_9a)

    regexp_9b = r'(?<=[/])NNS'
    temp_9b = re.findall(regexp_9b, comment['body'])
    feats_local[9-1]+=len(temp_9b)

    print('No. of common noun tokens: ' + str(feats_local[9-1]))

    #10 Count proper noun tokens
    temp_10a = list()
    temp_10b = list()

    regexp_10a = r'(?<=[/])NNP'
    temp_10a = re.findall(regexp_10a, comment['body'])
    feats_local[10-1]+=len(temp_10a)

    regexp_10b = r'(?<=[/])NNPS'
    temp_10b = re.findall(regexp_10b, comment['body'])
    feats_local[10-1]+=len(temp_10b) 

    print('No. of proper noun tokens: ' + str(feats_local[10-1]))

    #11 Count adverb tokens
    temp_10a = list()
    temp_10b = list()
    temp_10c = list()

    regexp_11a = r'(?<=[/])RB'
    temp_11a = re.findall(regexp_11a, comment['body'])
    feats_local[11-1]+=len(temp_11a)

    regexp_11b = r'(?<=[/])RBR'
    temp_11b = re.findall(regexp_11b, comment['body'])
    feats_local[11-1]+=len(temp_11b) 

    regexp_11c = r'(?<=[/])RBS'
    temp_11c = re.findall(regexp_11c, comment['body'])
    feats_local[11-1]+=len(temp_11c) 

    print('No. of adverb tokens: ' + str(feats_local[11-1]))

    #12 No. of wh- words
    temp_12 = ''

    regexp_12 = r'wh\S+(?=[/])'
    temp_12 = re.findall(regexp_12, comment['body'])
    feats_local[11]+=len(temp_12) 

    print('No. of wh- words: ' + str(feats_local[11]))

    #13 No. of slang words
    temp_13 = ''
    word = ''
    for i in slang_words:
        if (i == '\n'):
            regexp_13 = r' '+word+'(?=[/])'
            temp_13 = re.findall(regexp_13, comment['body'])
            feats_local[12]+=len(temp_13) 
            temp_13 = ''
            word = ''
        else:
            word += i

    print('No. of slang words: ' + str(feats_local[12]))

    #14 No. of words converted in uppercase
    # None since all words were converted to lowercase in the preprocessing step
    feats_local[13] = 0.0

    #15 Average length of sentences
    regexp_15 = '(.*)'
    temp_15 = re.findall(regexp_15, comment['body'])
    for i in temp_15:
        if len(i) != '':
            feats_local[14]+=len(i)/len(temp_15) 

    print('Avg. length of sentences: ' + str(feats_local[14]))

    #16 Average length of tokens
    regexp_16 = '\S+(?=[/])'
    temp_16 = re.findall(regexp_16, comment['body'])

    for i in temp_16:
        feats_local[15]+=len(i)/len(temp_16) 

    print('Avg. length of token: ' + str(feats_local[15]))

    #17 Calculate the no. of sentences
    regexp_17 = r'\n'
    temp_17 = re.findall(regexp_17, comment['body'])

    feats_local[16] = len(temp_17)
    print('No. of sentences: ' + str(len(temp_17)))

    #17-23 Calculate average lexical norms (Bristol)
    regexp_18 = r'\S+(?=[/])'
    temp_18 = re.findall(regexp_18, comment['body'])
    summer_18 = 0.0
    bristol = np.array(bristol)
    warringer = np.array(warringer)

    # Extract reqd. information from python object
    bristol_word = bristol[1:,0]
    bristol_aoa = bristol[1:,1]
    bristol_img = bristol[1:,2]
    bristol_fam = bristol[1:,3]
    ind_array = list()

    for i in temp_18:
        ind = np.where(i == bristol_word)
        if (ind[0].shape[0]):
            ind_array.append(ind[0][0])

    # Iniitialize calculation arrays
    br_aoa_temp = np.zeros(len(ind_array))
    br_img_temp = np.zeros(len(ind_array))
    br_fam_temp = np.zeros(len(ind_array))

    # Assign values
    br_aoa_temp = bristol_aoa[ind_array]
    br_img_temp = bristol_img[ind_array]
    br_fam_temp = bristol_fam[ind_array]

    # Assign mean values
    feats_local[17] = np.mean(br_aoa_temp.astype(np.float))
    feats_local[18] = np.mean(br_img_temp.astype(np.float))
    feats_local[19] = np.mean(br_fam_temp.astype(np.float))

    # Assign standard deviation
    feats_local[20] = np.std(br_aoa_temp.astype(np.float))
    feats_local[21] = np.std(br_img_temp.astype(np.float))
    feats_local[22] = np.std(br_fam_temp.astype(np.float))

    # 23-28 Calculate average lexical norms (Warringer)

    # Extract reqd. information from python object
    warringer_word = warringer[1:,0]
    warringer_v = warringer[1:,1]
    warringer_a = warringer[1:,2]
    warringer_d = warringer[1:,3]
    ind_array = list()

    for i in temp_18:
        ind = np.where(i == warringer_word)
        if (ind[0].shape[0]):
            ind_array.append(ind[0][0])

    # Iniitialize calculation arrays
    br_v_temp = np.zeros(len(ind_array))
    br_a_temp = np.zeros(len(ind_array))
    br_d_temp = np.zeros(len(ind_array))

    # Assign values
    br_v_temp = warringer_v[ind_array]
    br_a_temp = warringer_v[ind_array]
    br_d_temp = warringer_d[ind_array]

    # Assign mean values
    feats_local[23] = np.mean(br_v_temp.astype(np.float))
    feats_local[24] = np.mean(br_a_temp.astype(np.float))
    feats_local[25] = np.mean(br_d_temp.astype(np.float))

    # Assign standard deviation
    feats_local[26] = np.std(br_v_temp.astype(np.float))
    feats_local[27] = np.std(br_a_temp.astype(np.float))
    feats_local[28] = np.std(br_d_temp.astype(np.float)) 

    # Extract final data from files (30-173)
    if (comment['cat'] == "Left"):
        ind = np.where(comment['id']==np.array(left_id))
        feats_local[29:173] =  np.squeeze(left_data[ind])
        feats_local[173] = 0
    elif (comment['cat'] == "Center"):
        ind = np.where(comment['id']==np.array(center_id))
        feats_local[29:173] = np.squeeze(center_data[ind])
        feats_local[173] = 1
    elif (comment['cat'] == "Right"):
        ind = np.where(comment['id']==np.array(right_id))
        feats_local[29:173] = np.squeeze(right_data[ind])
        feats_local[173] = 2
    elif (comment['cat'] == "Alt"):
        ind = np.where(comment['id']==np.array(alt_id))
        feats_local[29:173] = np.squeeze(alt_data[ind])
        feats_local[173] = 3

    return feats_local

def main( args ):

    data = json.load(open(args.input))
    feats_1 = np.zeros( (len(data[0]), 173+1) )
    feats_2 = np.zeros( (len(data[0]), 173+1) )
    feats_3 = np.zeros( (len(data[0]), 173+1) )
    feats_4 = np.zeros( (len(data[0]), 173+1) )
    bristol = list()
    warringer = list()
    alt_id = list()
    right_id = list()
    left_id = list()
    center_id = list()
    alt_data = list()
    right_data = list()
    left_data = list()
    center_data = list()

    # bristol = open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv').read()
    # warringer = open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv').read()
    
    #slang_words = open('./Wordlists/Slang').read()
    slang_words = open('/u/cs401/Wordlists/Slang').read()
    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            bristol.append([row[1], row[3], row[4], row[5]])

    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            warringer.append([row[1], row[2], row[5], row[8]])

    with open('/u/cs401/A1/feats/Alt_IDs.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            alt_id.append(row[0])

    with open('/u/cs401/A1/feats/Center_IDs.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            center_id.append(row[0])

    with open('/u/cs401/A1/feats/Left_IDs.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            left_id.append(row[0])

    with open('/u/cs401/A1/feats/Right_IDs.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            right_id.append(row[0])

    alt_data = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')

    center_data = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')

    left_data = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')

    right_data = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')

    # print((np.array(data[0])).shape)
    # print(np.array(data[0][1000]))
    # print(np.array(data[0][1]))

    # Convert lists into numy arrays
    alt_id = np.array(alt_id)
    center_id = np.array(center_id)
    left_id = np.array(left_id)
    right_id = np.array(right_id)

    for i in range(len(data[0])):
        print('Extracting features [' + str(i) + '/' + str(len(data[0])) + ']')
        feats_1[i,:] = extract1(data[0][i], slang_words, bristol, warringer, alt_id, center_id, right_id, left_id, alt_data, center_data, right_data, left_data)
        print()

    for i in range(len(data[1])):
        print('Extracting features [' + str(i) + '/' + str(len(data[0])) + ']')
        feats_2[i,:] = extract1(data[1][i], slang_words, bristol, warringer, alt_id, center_id, right_id, left_id, alt_data, center_data, right_data, left_data)
        print()

    for i in range(len(data[2])):
        print('Extracting features [' + str(i) + '/' + str(len(data[0])) + ']')
        feats_3[i,:] = extract1(data[2][i], slang_words, bristol, warringer, alt_id, center_id, right_id, left_id, alt_data, center_data, right_data, left_data)
        print()

    for i in range(len(data[3])):
        print('Extracting features [' + str(i) + '/' + str(len(data[0])) + ']')
        feats_4[i,:] = extract1(data[3][i], slang_words, bristol, warringer, alt_id, center_id, right_id, left_id, alt_data, center_data, right_data, left_data)
        print()

    feats = np.concatenate((feats_1, feats_2, feats_3, feats_4),axis=0)

    np.savez_compressed( args.output, feats=feats )
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)

