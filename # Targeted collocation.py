# Importing required libraries
import re  # Regular Expression library for text cleaning
import nltk.corpus.reader.bnc as cor 
import pandas as pd  # Pandas for data manipulation
import numpy as np  # Numpy for numerical operations
import nltk 

baby_bnc=cor.BNCCorpusReader(root=r'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\download\\Texts',fileids=r'[a-z]{3}/\w*\.xml') 
bnc=cor.BNCCorpusReader(root=r'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\BNC\\download\\Texts',fileids=r'[A-Z]/\w*/\w*\.xml') 

#Read the corpus file and populate the 'tokens' list
collocation_textbook_corpus = 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Collocation Textbook Corpus.txt'


# Read the CSV file into a Pandas DataFrame
adj_noun_D8= 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Adjective -Noun combination - Discovery 8.csv'
adj_noun_D10 = 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Adjective Noun Combination - Discovery 10.csv'
adj_noun_D11  = 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Adjective-Noun Combination -Discovery 11.csv'
verb_noun_D8  = 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Verb Noun Combination - Discovery 8.csv'
verb_noun_D10  = 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Verb-Noun Combination Discovery 10.csv'
verb_noun_D11  = 'C:\\Users\\adria\\OneDrive - The Open University\\Loan_disert\\Dissertation - Minh Loan\\Words\\Verb Noun Combination - Discovery 11.csv'




def read_and_tokenize_corpus(corpus_file_path):
    """
    Reads a text file and tokenizes it into words.
    
    Parameters:
        corpus_file_path (str): The path to the text corpus file.
        
    Returns:
        list: A list containing all the tokens (words) in the text corpus.
    """
    tokens = []  # Initialize an empty list to store tokens
    with open(corpus_file_path, 'r', encoding='latin1') as file:
        corpus = file.read()  # Read the entire file
        tokens.extend(corpus.lower().split())  # Tokenize the text and append to tokens list
    return tokens

def clean_tokens(tokens):
    """
    Cleans a list of tokens by removing trailing punctuation.
    
    Parameters:
        tokens (list): A list of tokens (words) to clean.
        
    Returns:
        list: A list of cleaned tokens.
    """
    def clean_text(text):
        """
        Removes trailing punctuation from a given text.
        
        Parameters:
            text (str): The text to clean.
            
        Returns:
            str: The cleaned text.
        """
        return re.sub(r"([.,:\"/;'?!)()]+)$", '', text)
    
    return [clean_text(token) for token in tokens]  # Apply the cleaning function to each token

def read_targeted_collocations(csv_file_path):
    """
    Reads targeted collocations from a CSV file.
    
    Parameters:
        csv_file_path (str): The path to the CSV file containing targeted collocations.
        
    Returns:
        tuple: Two lists containing verb-noun and adjective-noun collocations.
    """
    df = pd.read_csv(csv_file_path)  # Read the CSV file into a DataFrame
    verb_noun = df.iloc[:, 0].tolist()  # Extract the second column containing verb-noun collocations
    #adj_noun = df.iloc[:, 2].tolist()  # Extract the third column containing adjective-noun collocations

    # Remove NaN elements from both lists
    verb_noun = [x for x in verb_noun if not (isinstance(x, (float, np.float64)) and np.isnan(x))]
    #adj_noun = [x for x in adj_noun if not (isinstance(x, (float, np.float64)) and np.isnan(x))]

    return verb_noun #adj_noun  # Return the two lists as a tuple

def corpus_bigram_textbook(list_of_words, corpus_tokens):
    """
    Computes the frequency of targeted bigrams in a corpus.
    
    Parameters:
        list_of_words (list): A list containing the targeted bigram strings (e.g., "run fast", "eat apple").
        corpus_tokens (list): A list of tokens (words) that make up the corpus.
        
    Returns:
        None: The function prints the frequency of each targeted bigram in the corpus.
    """
    # Create a list of bigrams from the corpus tokens
    bigrams = nltk.bigrams(corpus_tokens)
    
    # Compute the frequency distribution of all bigrams
    fdist = nltk.FreqDist(bigrams)
    
    # Initialize an empty list to store the targeted bigrams as tuples
    target_bigrams = []
    
    # Convert the list_of_words into tuples and store in target_bigrams
    for item in list_of_words:
        split_item = item.split()  # Split each item (bigram) into individual words

        # Check if the item has exactly two words, otherwise skip
        if len(split_item) != 2:
            print(f"Skipping '{item}' as it doesn't contain two words.")
            continue

        # Unpack the first and second words
        first_word, second_word = split_item
        
        # Create a tuple from the two words
        bigram_tuple = (first_word, second_word)
        
        # Add the tuple to the target_bigrams list
        target_bigrams.append(bigram_tuple)
        
    # Print the frequency of each targeted bigram
    for bigram in target_bigrams:
        print(f"{bigram}: {fdist[bigram]}")

def corpus_ngram_textbook_to_excel(list_of_words, corpus_tokens, output_filename, start_col, start_row):
    """
    Computes the frequency of targeted bigrams and trigrams in a corpus and writes the output to an Excel file.
    
    Parameters:
        list_of_words (list): A list containing the targeted n-gram strings (e.g., ["run fast", "eat apple", "go to school"]).
        corpus_tokens (list): A list of tokens (words) that make up the corpus.
        output_filename (str): The filename of the output Excel file.
        start_col (int): The column index where the new data will start in the Excel sheet.
        start_row (int): The row index where the new data will start in the Excel sheet.
        
    Returns:
        None: The function writes the output to an Excel file and prints frequency details to the console.
    """

    # Generate bigrams and trigrams from the corpus tokens
    bigrams = nltk.bigrams(corpus_tokens)
    trigrams = nltk.trigrams(corpus_tokens)
    
    # Compute frequency distribution for bigrams and trigrams
    fdist_bigram = nltk.FreqDist(bigrams)
    fdist_trigram = nltk.FreqDist(trigrams)
    
    # Initialize an empty list to store the data
    data = []
    
    # Iterate through the list of targeted words to find their frequencies
    for item in list_of_words:
        split_item = item.lower()
        split_item = split_item.split()  # Split each n-gram into individual words
        
        # Check for bigrams
        if len(split_item) == 2:
            first_word, second_word = split_item  # Unpack the first and second words
            bigram_tuple = (first_word, second_word)  # Create a tuple for the bigram
            #print(f"Bigram {bigram_tuple}: {fdist_bigram[bigram_tuple]}")  # Print the frequency
            data.append([' '.join(bigram_tuple), fdist_bigram[bigram_tuple], 'Bigram'])  # Append to data list
            
        # Check for trigrams
        elif len(split_item) == 3:
            first_word, second_word, third_word = split_item  # Unpack the first, second, and third words
            trigram_tuple = (first_word, second_word, third_word)  # Create a tuple for the trigram
            #print(f"Trigram {trigram_tuple}: {fdist_trigram[trigram_tuple]}")  # Print the frequency
            data.append([' '.join(trigram_tuple), fdist_trigram[trigram_tuple], 'Trigram'])  # Append to data list
        
        # Skip n-grams that are not bigrams or trigrams
        else:
            print(f"Skipping '{item}' as it doesn't contain two or three words.")
    
    # Convert the data list to a DataFrame
    df_new = pd.DataFrame(data, columns=['N-gram', 'Frequency', 'Type'])
    
    # Try to read existing data from the Excel file, if it exists
    try:
        with pd.ExcelFile(output_filename) as xls:
            df_old = pd.read_excel(xls, 'Sheet1')
    except FileNotFoundError:
        df_old = pd.DataFrame()  # Initialize an empty DataFrame if the file is not found
    
    # Combine the old and new DataFrames
    df_combined = pd.concat([df_old, df_new], axis=1)
    
    # Write the combined DataFrame to the Excel file
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        df_combined.to_excel(writer, sheet_name='Sheet1', startrow=start_row, startcol=start_col, index=False)

def filter_adjective_noun_pairs(word_pairs):
    """
    Filters a list of word pairs to return only those where the first word is an adjective 
    and the second word is a noun.

    Parameters:
    - word_pairs (list): A list of lists where each inner list contains a pair of words.

    Returns:
    - filtered_pairs (list): A list of lists where each inner list contains a pair of words
      where the first word is an adjective and the second is a noun.
    """
    
    # Initialize an empty list to store the filtered word pairs.
    filtered_pairs = []
    
    # Iterate over each pair in the list of word pairs.
    for pair in word_pairs:
        
        # Skip the pair if it does not contain exactly two words.
        if len(pair) != 2:
            continue
            
        # Unpack the first and second word from the pair.
        first_word, second_word = pair
        
        # Use NLTK's pos_tag function to get the part-of-speech tags for the words in the pair.
        pos_tags = nltk.pos_tag(pair)
        
        # Check if the first word is an adjective ('JJ', 'JJR', 'JJS') 
        # and the second word is a noun ('NN', 'NNS', 'NNP', 'NNPS').
        if pos_tags[0][1] in ['JJ', 'JJR', 'JJS'] and pos_tags[1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            
            # If both conditions are met, add the pair to the list of filtered pairs.
            filtered_pairs.append(pair)
    
    # Return the list of filtered pairs.
    return filtered_pairs

def filter_verb_third_noun_triplets(word_triplets):
    """
    Filters a list of word triplets to return only those where the first word is a verb 
    and the third word is a noun.

    Parameters:
    - word_triplets (list): A list of lists where each inner list contains a triplet of words.

    Returns:
    - filtered_triplets (list): A list of lists where each inner list contains a triplet of words,
      the first word being a verb and the third word being a noun.
    """
    
    # Initialize an empty list to store the filtered word triplets.
    filtered_triplets = []
    
    # Iterate over each triplet in the list of word triplets.
    for triplet in word_triplets:
        
        # Skip the triplet if it does not contain exactly three words.
        if len(triplet) != 3:
            continue
            
        # Use NLTK's pos_tag function to get the part-of-speech tags for the words in the triplet.
        pos_tags = nltk.pos_tag(triplet)
        
        # Check if the first word is a verb ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ') 
        # and the third word is a noun ('NN', 'NNS', 'NNP', 'NNPS').
        if pos_tags[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and pos_tags[2][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            
            # If both conditions are met, add the triplet to the list of filtered triplets.
            filtered_triplets.append(triplet)
    
    # Return the list of filtered triplets.
    return filtered_triplets

def filter_verb_noun_pairs(word_pairs):
    """
    Filters a list of word pairs to return only those where the first word is a verb 
    and the second word is a noun.

    Parameters:
    - word_pairs (list): A list of lists where each inner list contains a pair of words.

    Returns:
    - filtered_pairs (list): A list of lists where each inner list contains a pair of words,
      the first word being a verb and the second being a noun.
    """
    
    # Initialize an empty list to store the filtered word pairs.
    filtered_pairs = []
    
    # Iterate over each pair in the list of word pairs.
    for pair in word_pairs:
        
        # Skip the pair if it does not contain exactly two words.
        if len(pair) != 2:
            continue
            
        # Unpack the first and second word from the pair.
        first_word, second_word = pair
        
        # Use NLTK's pos_tag function to get the part-of-speech tags for the words in the pair.
        pos_tags = nltk.pos_tag(pair)
        
        # Check if the first word is a verb ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ') 
        # and the second word is a noun ('NN', 'NNS', 'NNP', 'NNPS').
        if pos_tags[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and pos_tags[1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            
            # If both conditions are met, add the pair to the list of filtered pairs.
            filtered_pairs.append([first_word, second_word])
    
    # Return the list of filtered pairs.
    return filtered_pairs

def find_freq(corpus, word_list, excel_filename="word_frequencies.xlsx"):
    """
    Function to find the frequency of the first and last word in each phrase
    in a given word list and saves the result in an Excel file.
    
    Parameters:
    - corpus (list): List of words to create the frequency distribution from.
    - word_list (list): List of phrases to find the first and last word frequencies for.
    - excel_filename (str, optional): Name of the Excel file to save the results in.
                                      Default is 'word_frequencies.xlsx'.
    
    Returns:
    None. Writes results to an Excel file.
    """

    # Convert the corpus to lowercase for case-insensitive comparison
    corpus = [word.lower() for word in corpus]
    
    # Create a frequency distribution from the corpus
    freq_dist = nltk.FreqDist(corpus)
    
    # Initialize an empty list to hold the result dictionaries
    result = []
    
    # Loop through each phrase in the word_list
    for words in word_list:
        
        # Convert the phrase to lowercase and split into a list of words
        words_split = words.lower().split()
        
        # Extract the first and last word from the list
        first_word = words_split[0]
        last_word = words_split[-1]
        
        # Find the frequency of the first and last word in the corpus
        first_word_freq = freq_dist[first_word]
        last_word_freq = freq_dist[last_word]
        
        # Append a dictionary of the result to the result list
        result.append({
            "Phrase": words,
            "First_Word": first_word,
            "First_Word_Frequency": first_word_freq,
            "Last_Word": last_word,
            "Last_Word_Frequency": last_word_freq
        })
    
    # Convert the result list to a Pandas DataFrame
    df = pd.DataFrame(result)
    
    # Write the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)

