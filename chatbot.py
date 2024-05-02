import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
import spacy
from spacy.matcher import Matcher
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
pd.set_option('display.max_colwidth', 200)
# nltk.download('punkt')
model = SentenceTransformer('all-mpnet-base-v2')

# Function to scrape text from a webpage and split into sentences
def scrape_and_split_text(url, max_words=18):
    """Scrapes text from a webpage, splits into sentences, and breaks long sentences."""

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=' ')

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Break long sentences
        split_sentences = []
        for sentence in sentences:
            while len(sentence.split()) > max_words:
                split_index = sentence.rfind(' ', 0, max_words)
                if split_index == -1:  # Handle cases where no suitable space is found
                    split_index = max_words
                split_sentences.append(sentence[:split_index].strip())
                sentence = sentence[split_index:].strip()
            split_sentences.append(sentence.strip())

        return split_sentences

    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        return None

# Function to get entities from a sentence
def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################
    return [ent1.strip(), ent2.strip()]

# Function to get the relation between the entities
def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    #define the pattern
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    # print(matches)
    k = len(matches) - 1

    # Error Handling
    if not matches:
      return None  # Or return a default value like "" or "NO_RELATION_FOUND"

    # Only proceed if there's at least one match
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]]  # This should now be safe

    return(span.text)

# Function to check sentence length
def is_short_sentence(sentence):
    words = sentence.split()  # Split into words
    return len(words) <= 2

# Function to generate response for the query using the Knowledge Graph
def generate_response(query):
    # open json file and load the filtered DataFrame
    kg_df_filtered = pd.read_json("kg_df_filtered.json")
    best=""
    max_score=-1
    query_embedding = model.encode(query, convert_to_tensor=True)
    for index, row in kg_df_filtered.iterrows():
        source_embedding = model.encode(row['source'], convert_to_tensor=True)
        target_embedding = model.encode(row['target'], convert_to_tensor=True)
        source_similarity = util.cos_sim(query_embedding, source_embedding)
        target_similarity = util.cos_sim(query_embedding, target_embedding)
        if max(source_similarity, target_similarity) > max_score:
            best = f"{row['source']} {row['edge']} {row['target']}"
            max_score = max(source_similarity, target_similarity)
    return best

# Function to chat with the bot on command line
def chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input)
        print("Bot:", response)

# Function to chat with the bot on streamlit
def helper(req):
    response = generate_response(req)
    return response


def main():
    url = "https://www.hdfcergo.com/health-insurance/individual-health-insurance/"
    input_text = scrape_and_split_text(url)
    df = pd.Series((v for v in input_text))
    
    # extract entities
    entity_pairs = []
    for i in tqdm(df):
        entity_pairs.append(get_entities(i))
    
    # extract relations
    relations = [get_relation(i) for i in df]

    # extract source
    source = [i[0] for i in entity_pairs]

    # extract target
    target = [i[1] for i in entity_pairs]

    # create a dataframe
    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    kg_df.sort_values(['source'], ascending=[True])

    # create a directed-graph from a dataframe
    G=nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    # plot the graph
    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='red', edge_cmap=plt.cm.Blues, pos = pos)
    # plt.show()
    # save the graph as an image
    plt.savefig("graph.png")


    # Filter the DataFrame
    column_to_check = 'target'  # Replace with the actual column name
    kg_df_filtered = kg_df[~kg_df[column_to_check].apply(is_short_sentence)]

    # store the filtered DataFrame in a json
    kg_df_filtered.to_json("kg_df_filtered.json")


if __name__ == "__main__":
    main()
    # chat()

