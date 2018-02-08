import sys
import argparse
import os
import json
import re
import spacy
import html
import string
import numpy as np

indir = '/u/cs401/A1/data/';
#indir = './data/'
abbr_words = open('/u/cs401/Wordlists/abbrev.english','r').read()
#abbr_words = './abbrev.english'

def preproc1( comment, stop_list, steps=range(1,11)):
	''' This function pre-processes a single comment

	Parameters:
		comment : string, the body of a comment
		steps   : list of ints, each entry in this list corresponds to a preprocessing step

	Returns:
		modComm : string, the modified comment
	'''

	# Initialize reqd. spacy variables
	nlp = spacy.load('en', disable=['parser', 'ner'])
	#nlp = spacy.load('en_core_web_lg')

	modComm = ''
	if 1 in steps:
		# Remove newline characters from data
		modComm = re.sub("\n", "", comment)
		#print(modComm)
	if 2 in steps:
		# Convert HTML codes to ASCII
		modComm = html.unescape(modComm)
		#print(modComm)
	if 3 in steps:
		# Remove links from comment
		modComm = re.sub(r'http\S+', '', modComm)
		modComm = re.sub(r'www\S+', '', modComm)
		#print(modComm)
	if 4 in steps:
		# Add padding around punctuation
		new_string = ''
		for i,j in enumerate(modComm):
			#print(j)
			if (i==len(modComm)-1):
				current = modComm[i]
				next = ''
				previous = modComm[i-1]
			elif (i==0):
				current = modComm[i]
				next = modComm[i+1]
				previous = ''
			else:
				current = modComm[i]
				next = modComm[i+1]
				previous = modComm[i-1]

			if (j in string.punctuation and j != "'" and j != "." and not(next in string.punctuation) and not(previous in string.punctuation)): # exclude apostrophes
				new_string += ' '+j
				if (next != ' '):
					new_string += ' '
			else:
				new_string += j

		modComm = new_string

		#print('After punctuation padding')
		#print(modComm)

	if 5 in steps:
		# Separating clitics (n't and 's)
		modComm = re.sub(r"n't", " n't", modComm)
		modComm = re.sub(r"'s", " 's", modComm)

		# print('After clitics ...')
		# print(modComm)
		# print()

	if 6 in steps:
		# Tagging each of the token
		utterance = nlp(modComm)
		modComm_new = ''
		#print(utterance)
		for token in utterance:
			modComm_new += token.text
			modComm_new += '/'
			modComm_new += token.tag_ + ' '
		modComm = modComm_new

		#print('After tagging ...')
		#print(modComm)
		#print()

	if 7 in steps:
		# Remove stop words
		modComm_new = modComm

		# Iteratively remove the stop words from the current string
		for i in stop_list:
			#i = re.sub('\n','',i)
			rem = r' '+i+'(?=[/])\S+'
			modComm_new = re.sub(rem, '', modComm_new)

		# Remove rogue forward slashes
		modComm_new = re.sub(r' / ', '', modComm_new)  
		modComm = modComm_new

		# print('After removing stop words ...')
		# print(modComm)
		# print()

	if 8 in steps:
		# Lemmatization:

		# Do a positive lookahead and find all of the instances of the word we are looking for
		rem = r'\S+(?=[/])'
		modComm_new = re.findall(rem, modComm)
		lemma_string = ''
		modComm_temp = ''

		# print('Before lemmatization (list) ...')
		# print(modComm_new)
		# print()

		# Convert the list of remaining words into a string to allow for lemma processing 
		for item in modComm_new:
			lemma_string += item
			lemma_string += ' '

		# print('Before lemmatization (string) ...')
		# print(lemma_string)
		# print()

		# Tagging each of the tokens
		utterance = nlp(lemma_string)

		# Attempt the lemmatization and tagging
		for token in utterance:
			modComm_temp += token.lemma_
			modComm_temp += '/'
			modComm_temp += token.tag_ + ' '

		modComm = modComm_temp
		# print('After lemmatization ...')
		# print(modComm)
		# print()

	if 9 in steps:
		modComm_new = modComm

		# Replace each period that corresponds to an abbreviation with a placeholder
		# for i in abbr_list:
		regexpy1 = ' \./\. '
		modComm_new = re.sub(regexpy1, '\n', modComm_new)

		
		# print('After putting in placeholder...')
		# print(modComm_new)
		# print()

		modComm = modComm_new

	if 10 in steps:
		# lowercase applied in earlier step!

		# Remove rogue forward slashes (if they exist!)
		modComm = re.sub(r' / ', '', modComm) 
		
		# print('After removing rogue...')
		# print(modComm)
		# print()		

	return modComm

def main( args ):

	allOutput = []
	for subdir, dirs, files in os.walk(indir):
		for file in files:
			fullFile = os.path.join(subdir, file)
			print("Processing " + fullFile)

			data = json.load(open(fullFile))

			# initialize variables
			new_array = dict()
			new_list = list()

			# Extract stop words
			stop_words = open('/u/cs401/Wordlists/StopWords','r').read()
			#stop_words = open('./StopWords','r').read()
			stop_list = list()
			new_word = ''
			for i in stop_words:
				if (i == '\n'):
					stop_list.append(new_word)
					new_word = ''
				else:
					new_word += i

			#print(args.ID)
			#print(len(data))
			for i in range(int(args.ID[0]%len(data)),len(data)):
				row = json.loads(data[i]) # TODO: read those lines with something like `j = json.loads(line)`
				if ("ups" in row and "downs" in row and row["author"] != "[deleted]"): # TODO: select appropriate args.max lines
					print('Processing comment ' + str(len(new_list)+1) + ' of 10,000 ...')
					new_array = dict()
					new_array["ups"] = row["ups"] # TODO: choose to retain fields from those lines that are relevant to you
					new_array["downs"] = row["downs"]
					new_array["score"] = row["score"]
					new_array["controversiality"] = row["controversiality"]
					new_array["subreddit"] = row["subreddit"]
					new_array["author"] = row["author"]
					new_array["body"] = preproc1(row["body"], stop_list)
					new_array["id"] = row["id"]
					new_array["cat"] = file # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 

					new_list.append(new_array)
				if (len(new_list)==args.max):
					break;

			allOutput.append(new_list)

	fout = open(args.output, 'w')
	fout.write(json.dumps(allOutput))
	fout.close()

	return allOutput

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Process each .')
	parser.add_argument('ID', metavar='N', type=int, nargs=1,
						help='your student ID')
	parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
	parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
	args = parser.parse_args()

	if (args.max > 200272):
		print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
		sys.exit(1)

	# Extract stop_list
	# stop_words = open('/u/cs401/Wordlists/StopWords','r').read()
	# #stop_words = open('./StopWords','r').read()
	# stop_list = list()
	# new_word = ''
	# for i in stop_words:
	# 	if (i == '\n'):
	# 		stop_list.append(new_word)
	# 		new_word = ''
	# 	else:
	# 		new_word += i

	main(args)

	# Test
	#stringy = 'https://whatitgoing !! \www\.gmail.cm &#33; /////// plays\' spam-egg yoyomab hakuna\'s matata\'s'
	#stringy = 'house CHAMILLIONAIRE user? ways'
	#stringy = """ e.g. i.e. Blvd. Ave. To be fair e.g. i.e. : it is a bit suspect to count Nigeria as a Muslim country , as this is a divided nation and only the northern half  ( which also tends to be poorer ) is Islamic. It is also worth noting that Turkey doesn't appear on the list... I don't know why. Of ocurse , it is offiically secular , but so is Georgia where the Boston Marathon bombers came from. I think , as a whole , Muslim student visas are even a massive issue because of the number that bring their wives over during the 7th month of pregnancy , get them the 3 month visitor's visa and... voila. We now have a new  " American."I have met some Koreans my age who are 'American citizens' who cannot speak English well and have never lived there for a prolonged time , but have dodged mandatory Korean military service and have the right to vote in our elections because they cheated the system. The system the US has is so confusing and ridiculous that I am even questioned on these policies by foreigners  ( I live abroad), and when I explain it to them they are shocked and ask rapid questions , in disbelief."""
	#print(preproc1(stringy, stop_list))


