import folia.main as folia
from collections import Counter
import re
import glob
import os.path
import sys
import numpy as np
import xml.etree.ElementTree as ET
import csv

# Let users choose between A, corresponding to HitPaRank's modes
# F and E (ngram frequency lists and expansions) or
# B, corresponding to HitPaRank's mode A (paragraph extraction).

while True:
    a_or_b = input(
        'Choose A or B. I want to:\n'
        'A\tGenerate 1-4 gram frequency tsv files and expand '
        '(wildcarded) ngrams from a FoLiA corpus.\n'
        'B\tExtract paragraphs from my FoLiA corpus using '
        'files I previously obtained by doing A.\n'
        'Or, you can exit HitPyRank by entering \'exit\'. '
    ).lower()
    if a_or_b not in ['a', 'b', 'exit']:
        print(
            'That was \'A\' nor \'B\' nor \'exit\', '
            'you are kindly invited to try again.'
        )
    else:
        break

if a_or_b == 'a':

    # Mode A (ngram, lemma + frequency extraction, and expansion)
    ns = {"folia": "http://ilk.uvt.nl/folia"}

    # Function to generate ngrams from a list of words
    def generate_ngrams(words, n):
        if n == 1:
            return [(word.lower(),) for word in words]
        else:
            return [tuple(word.lower() for word in words[i:i + n]) for i in
                    range(len(words) - n + 1)]

    # Mode A (ngram + frequency extraction, and ngram expansion)
    # Ask for path to FoLiA corpus
    corpus_folder = input(
        'Specify the location of your FoLiA corpus.\n'
        'Note that all xml files in your specified path\n'
        'will be processed by HitPyRank, so please make \n'
        'sure that all and only the xml files you want to \n'
        'extract 1-4 grams from are there!\n '
    )

    # Initialize counters
    word_lemma_counter = Counter()
    ngram_counter = Counter()

    # Process each XML file in the corpus folder
    for filename in glob.glob(os.path.join(corpus_folder, '*.xml')):
        print('Processing:', filename)

        tree = ET.parse(filename)
        root = tree.getroot()

        # Extract words and lemmas
        word_elements = root.findall(".//folia:w", namespaces=ns)
        word_lemma_list = []
        for word_elem in word_elements:
            text_elem = word_elem.find("folia:t", namespaces=ns)
            lemma_elem = word_elem.find("folia:lemma", namespaces=ns)
            if text_elem is not None and lemma_elem is not None and "class" in lemma_elem.attrib:
                word_text = text_elem.text.strip().lower()
                lemma_class = lemma_elem.attrib["class"].lower()
                word_lemma_list.append((word_text, lemma_class))

        # Update word-lemma counter
        word_lemma_counter.update(word_lemma_list)

        # Extract ngrams
        sentence_elements = root.findall(".//folia:s", namespaces=ns)
        for sentence_elem in sentence_elements:
            word_elements = sentence_elem.findall(".//folia:w", namespaces=ns)
            words_list = [word_elem.find("folia:t", namespaces=ns).text.strip() for word_elem in word_elements]
            ngrams = generate_ngrams(
                words_list, 1) + generate_ngrams(
                words_list, 2) + generate_ngrams(
                words_list, 3) + generate_ngrams(
                words_list, 4
            )
            ngram_counter.update(ngrams)

    save_ask = input(f'\nDo you want to write the 1-4 gram word frequency tsv\n files to {os.getcwd()}? (Y/N)').lower()
    while True:
        if save_ask == 'y':
            # Save ngrams and frequencies to a TSV file
            ngram_tsv_filename = "ngrams_frequencies.tsv"
            with open(ngram_tsv_filename, "w") as ngram_tsv_file:
                for ngram, frequency in ngram_counter.items():
                    ngram_str = " ".join(ngram)
                    ngram_tsv_file.write(f"{ngram_str}\t{frequency}\n")

            print(f"Ngrams and frequencies saved to {ngram_tsv_filename}")

            # Save word-lemma pairs and frequencies to a TSV file
            word_lemma_tsv_filename = "word_lemma_frequencies.tsv"
            with open(word_lemma_tsv_filename, "w") as word_lemma_tsv_file:
                for (word, lemma), frequency in word_lemma_counter.items():
                    word_lemma_tsv_file.write(f"{word}\t{frequency}\t{lemma}\n")

            print(f"Word-lemma pairs and frequencies saved to {word_lemma_tsv_filename}")
            break
        elif save_ask == 'n':
            print('Ok!')
            break
        else:
            print('That was Y nor N, please try again. ')

    # End of word/lemma + freq extraction
    # Begin word/lemma expansion

    # Ask for no. of lists (1-20)
    with open('HitPyRank.ModeA.ListExample.tsv', 'w') as f:
        f.write(
            'unigram\t2\tW\nnon-wildcarded bigram\t3\tW\nwildcarded bigram*\t2\tL\na wildcarded trigram*\t1\tL')
    print('Now you are about to load your tsv\'s with ngrams for expansion.\n'
          'Before you continue, have a look at the tsv file called:\n'
          '\'HitPyRank.ModeA.ListExample.tsv\'/n '
          'which was just written to '
          f'{os.getcwd()}.\n'
          'As you can see there, your input tsv should not contain headers\n'
          'and should contain entries in 3 tab-separated columns, namely:\n'
          'ngram\trank\tword/lemma\n\n'
          'In the first column you list your ngrams, which will serve as\n'
          'search terms. You can expand ngrams up to 4-grams, and you can\n'
          'indicate that you want to expand terms by appending ngrams with\n'
          'an asterisk (e.g. \'know*\' will expand to \'knowledge\' and \'unbeknownst\').\n\n'
          'In the second column you specify an ngram\'s rank (0-3).\n'
          'Note that entries ranked \'0\' will be expanded (because its\n'
          'expansions could be ranked higher) but will not be used to\n'
          'retrieve paragraphs with in Mode B.\n\n'
          'In the third column you indicate with \'w\' or \'l\' whether \n'
          'you want HitPyRank to expand the lemma or the word form of your ngram.\n'
          'If you enter anything other than \'w\' or \'l\' (case-insensitive) \n'
          'in this column, HitPyRank will default to \'w\'. ')

    while True:
        number_of_lists = input(
            '\nHow many lists of terms will you be using?\n'
            '(Only positive integers, min. 1 and max. 20) \n')
        try:
            val = int(number_of_lists)
            if val < 1 or val > 20:
                print('Sorry, input must a positive integer between 1 and 20. ')
                continue
            break
        except ValueError:
            print('That\'s not an integer! ')

    if int(number_of_lists) == 1:
        print("\nYou will be using 1 list. \n")
    else:
        print('\nYou will be using ' + number_of_lists + ' lists.\n')

    # Ask for (wildcarded) ngram lists
    input_tsv_paths = []
    for n in range(0, int(number_of_lists)):
        while True:
            input_tsv_path = input(f'Please specify the path to your '
                                   f'tsv file for List {n+1}. ')
            if not input_tsv_path.endswith('.tsv'):
                print('That\'s not a tsv file!')
            elif not os.path.isfile(input_tsv_path):
                print('Cannot find that tsv file on the specified path.')
            elif input_tsv_path in input_tsv_paths:
                print('You have already selected this tsv file,\n'
                      'there is no use in using the same list multiple times.\n'
                      'Please select another list.')
            else:
                break
        input_tsv_paths.append(input_tsv_path)

    # Check if ngram and lemma lists are still in place
    while True:
        list_moved = input(f'\nHave you moved \'ngrams_frequencies.tsv\' or \'word_lemma_frequencies.tsv\'\n'
                           f'from {os.getcwd()}? (Y/N)\n').lower()
        if list_moved != 'n' and list_moved != 'y':
            print('\nThat was Y nor N, please try again.\n')
            continue
        elif list_moved == 'y':
            ngram_path = input('\nPlease specify the path to \'ngrams_frequencies.tsv\''
                               ' and \'word_lemma_frequencies.tsv\'\n'
                               '(they ought to be in the same folder).\n')
            break
        else:
            ngram_path = os.getcwd()
            break

    print(f'\n\'ngrams_frequencies.tsv\' and \'word_lemma_frequencies.tsv\' in {ngram_path}\n')

    expansions_lists = []

    def expansion_writer(inputrow, ngramrow, filepath):
        """Appends 6 items to expansions_list:
        input ngram, expansion, freq, rank, wl, listname"""
        expansions_lists.append(inputrow[0] + '\t' +  # ngram
                                str(ngramrow[0]) + '\t' +  # expansion
                                str(ngramrow[1]) + '\t' +  # freq
                                inputrow[1] + '\t' +  # rank
                                inputrow[2] + '\t' +  # wl
                                filepath[filepath.rfind('/') + 1:].replace('.tsv', '')  # listname
                                )

    print('loading ngrams to memory')
    ngram_dict = {}
    with open(f'{ngram_path}/ngrams_frequencies.tsv', 'r', newline='') as ngrams_file:
        ngrams_tsv = csv.reader(ngrams_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in ngrams_tsv:
            ngram_dict[row[0]] = row[1]
    print('done loading\n\n')

    print('loading lemmas/unigrams to memory')
    lemma_dict = {}
    with open(f'{ngram_path}/word_lemma_frequencies.tsv', 'r', newline='') as lemmas_file:
        lemmas_tsv = csv.reader(lemmas_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in lemmas_tsv:
            lemma_dict.setdefault(row[2], []).append([row[0], row[1]])

    print('done loading\n\n'
          'expanding your terms..\n')

    for path in input_tsv_paths:
        term_rank_wl_array = np.genfromtxt(fname=path,
                                           delimiter="\t",
                                           skip_header=0,
                                           filling_values='n/a',
                                           dtype='str',
                                           ndmin=2)
        for irow in term_rank_wl_array:
            term = irow[0]
            freq = irow[1]
            wl_type = irow[2].lower()

            if wl_type == 'l':   # lemmas
                if ' ' in term:
                    raise Exception('There is an n-gram in one of your input lists\n'
                                    'for which n > 1 and type = lemma (\'L\'). \n'
                                    'You can only use unigram lemmas. Please adjust your input list.')
                elif '*' in term:
                    raise Exception('You have a wildcarded lemma in one of your input lists.\n'
                                    'An input lemma should be a non-wildcarded unigram, \n'
                                    'which will be expanded to all word forms with that \n'
                                    'lemma tag in your corpus. Please adjust your input list. ')
                else:
                    for expansion in lemma_dict.get(term, []):
                        expansion_writer(irow, expansion, path)
            else:  # Wildcarded words (not lemmas)
                if '*' in term:
                    wildcard_pattern = '.*' + term.replace('*', '.*')
                    for ngram, ngram_freq in ngram_dict.items():
                        if re.match(wildcard_pattern, ngram) and term.count(' ') == ngram.count(' '):
                            expansion_writer(irow, [ngram, ngram_freq], path)
                else:  # No wildcards
                    if term in ngram_dict:
                        expansion_writer(irow, [term, ngram_dict[term]], path)

    print('done expanding terms\n\n')

    list_names = set()

    for line in expansions_lists:
        list_name = line.split('\t')[-1]
        list_names.add(list_name)

    for list_name in list_names:
        with open(list_name + '.expanded.tsv', 'w') as output_tsv:
            for line in expansions_lists:
                if line.split('\t')[-1] == list_name:
                    output_tsv.write(line + '\n')
    print('Expansions written to ' + os.getcwd() + '.')
# End of mode A

# Mode B: paragraph extraction
elif a_or_b == 'b':
    # Ask for no. of lists (1-20)
    while True:
        number_of_lists_B = input(
            f'How many lists of (expanded) terms will you be using?\n'
            f'(Only positive integers, min. 1 and max. 20)')
        try:
            val_B = int(number_of_lists_B)
            if val_B < 1 or val_B > 20:
                print('Sorry, input must a positive integer between 1 and 20. ')
                continue
            break
        except ValueError:
            print('That\'s not an integer! ')

    if int(number_of_lists_B) == 1:
        print("You will be using 1 list.")
    else:
        print('You will be using ' + number_of_lists_B + ' lists.')

    # Ask for (wildcarded) ngram lists
    input_tsv_path_list_B = []
    for n in range(0, int(number_of_lists_B)):
        while True:
            input_tsv_path_B = input(
                f'Please specify the path to your tsv file for List {n+1}. '
                f'Make sure you specify lists of the following form: '
                f'\'/path/[List Name].expanded.tsv\'. ')
            try:
                if not input_tsv_path_B.endswith('.tsv'):  # checks if file is a tsv file
                    print('That\'s not a tsv file! ')
                    continue
                elif not os.path.isfile(input_tsv_path_B):  # checks if tsv file exists on specified location
                    print('Cannot find that tsv file on the specified path')
                    continue
                elif input_tsv_path_B in input_tsv_path_list_B:  # checks if tsv file is input twice
                    print(
                        'You have already selected this tsv file, there is no use in using '
                        'the same list multiple times. Please select another list. ')
                    continue
                break
            except:
                ('Something went wrong')
        input_tsv_path_list_B.append(input_tsv_path_B)

    input_expanded_lists_B = []  # list of lists with 3 items: input term, rank, listname

    for path in input_tsv_path_list_B:
        with open(path, 'r', newline='', encoding='utf-8') as input_file:
            reader = csv.reader(input_file, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                expansion = row[1].strip()
                rank = row[3].strip()
                listname = path[path.rfind("/")+1:].replace(".expanded.tsv", '')
                input_expanded_lists_B.append([expansion, rank, listname])

    def exact_matcher(term):
        """Turns input term into a regular expression with word boundaries for exact matching"""
        r = r'\b' + re.escape(term.lower()) + r'\b'
        return r


    corpus_folder_B = (input('Specify the location of your FoLiA corpus.\n'
                             'Note that all xml files in your specified path\n'
                             'will be processed by HitPyRank, so please make sure\n'
                             'that all and only the xml files you want to\n'
                             'extract paragraphs from are there!'))

    # Build paragraph dict (for each p.id: hits, counts, list names, and the paragraph itself)
    paragraph_dict = {}
    for filename_B in glob.glob(os.path.join(corpus_folder_B, '*.xml')):
        print('Processing:', filename_B)
        doc = folia.Document(file=filename_B)

        for p in doc.paragraphs():
            str_p = str(p)
            p_id = p.id
            for item in input_expanded_lists_B:
                if re.search(exact_matcher(item[0]), str_p.lower()):
                    paragraph_data = paragraph_dict.setdefault(p_id, {'paragraph': str_p, 'hit_terms': {}})
                    hit_terms = paragraph_data['hit_terms']
                    hit_term_data = hit_terms.setdefault(item[0], {'List': item[2], 'Hit count': 0, 'Rank': item[1]})
                    hit_term_data['Hit count'] += len(re.findall(exact_matcher(item[0]), str_p.lower()))

    # Build summaries
    # Calculate no. of digits in generic paragraph IDs
    num_paragraphs = len(paragraph_dict)
    num_zeros = len(str(num_paragraphs))

    for i, (p_id, p_data) in enumerate(paragraph_dict.items(), start=1):
        unique_summary = {}
        full_summary = []
        unique_lists = set()

        if 'hit_terms' in p_data:
            for term, term_data in p_data['hit_terms'].items():
                # Unique hit summary
                rank = term_data['Rank']
                term_key = f"{term_data['List']}:{rank}"
                unique_summary[term_key] = unique_summary.get(term_key, 0) + 1

                # Full summary
                full_summary.append(f"{term}.{rank}.{term_data['List']}.{term_data['Hit count']}")

                # Labels (unique list names)
                unique_lists.add(term_data['List'])

        # Unique hit summary & full summary
        paragraph_dict[p_id]['Unique hit summary'] = [f"{key}:{value}" for key, value in unique_summary.items()]
        paragraph_dict[p_id]['Full summary (term.rank.list.count)'] = full_summary

        # Assign label (equivalent of HitPaRank's NNYNY labels)
        label_value = ''.join(sorted(unique_lists))
        paragraph_dict[p_id]['Label'] = label_value

        # Unique generic ID's (e.g. p_0000234)
        paragraph_id_value = f"p_{str(i).zfill(num_zeros)}"
        paragraph_dict[p_id]['Paragraph ID'] = paragraph_id_value

    # Save tsv
    headers = ['Paragraph ID',
               'p.id',
               'Label',
               'Full summary (term.rank.list.count)',
               'Unique hit summary', 'paragraph'
               ]
    tsv_file = "output.tsv"

    with open(tsv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        for p_id, p_data in paragraph_dict.items():
            row_data = {
                'Paragraph ID': p_data.get('Paragraph ID', ''),
                'p.id': p_id,
                'Label': p_data.get('Label', ''),
                'Full summary (term.rank.list.count)': '\t'.join(p_data.get('Full summary (term.rank.list.count)', [])),
                'Unique hit summary': '\t'.join(p_data.get('Unique hit summary', [])),
                'paragraph': p_data.get('paragraph', '')
            }
            writer.writerow(row_data)

    print(f"TSV file '{tsv_file}' has been created successfully.")

else:
    sys.exit('Have a good day!')

# Wishlist:
# enable POS specification: write pos for each expansion
# make sure input corpus exists and is folia/xml
# enhance console prints for readability
# check if wildcarded terms appear in the middle of search terms
# review and rewrite all printed text
# test with faulty lists, and see if you can warn for or fix errors. In particular:
#  # input ngram is a >4 gram
#  # ranking is <0 or >3
#  # wildcards appear in the middle of words
#  # numbers or special characters are in input terms
#  # rankings are letters or special characters

# Make sure at mode A (of PYrank) something happens when you enter nonexisting path, or
# when there are no FoLiA files

# If you input the same input list multiple times (say you choose 3 lists, and you input a.tsv 3 times)
# it only outputs a correct expansion tsv once, and the rest are empty tsv's
# (does it? perhaps an older version of this script)

# Does 20 input lists really work? Did not test

# strip trailing/leading spaces in inputlists, but NOT spaces in between >1 grams

# limit Ranking from 0-3 (=4 will not be read (or output error) and 0 will be read and expanded,
# but will not be used to retrieve paragraphs)

# Explain what the expanded files should look like, just like the explanation in mode A

# Explain in which order lists must be input

# Do a scan on input lists: only alphabeticals and spaces allowed

# Maybe some new metrics?
#  Tersity: a score, a paragraph scores high if highly ranked terms are 1) many and 2) are close (in terms of space),
#  and 3) are not surrounded by many irrelevant words.

# if an empty corpus folder is given, tell! (now it just goes on to ask to write tsv)
