# Here we do **extractive summarization**

# **steps:**
# 1. text cleaning
# 2. word tokenization
# 3. word frequency table
# 4. sentence tokenization
# 5. summarization
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import PyPDF2
import requests
from bs4 import BeautifulSoup
nlp = spacy.load('en_core_web_sm')

pdf_file = open(r' ','rb')

# Create a PDF reader
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Initialize the text variable
text =  """
Text
     
 """

# Loop through each page in the PDF file
for page_num in range(len(pdf_reader.pages)):
    # Get the page and extract the text
    page = pdf_reader.pages[page_num]
    text += page.extract_text()

# Close the PDF file
pdf_file.close()
def extract_text_from_website (url) :
    predefined_text = """
    Text
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    website_text = ' '.join([p.text for p in soup.find_all('p')])
    # Combine predefined text and website text
    combined_text = predefined_text.strip() + "\n\n" + website_text.strip()
    
    return combined_text

# Example usage:
text = extract_text_from_website("https://en.wikipedia.org/wiki/Main_Page")
print(text)
   
text = '''YOU don’t know about me without you have read a book
by the name of The Adventures of Tom Sawyer; but that
ain’t no matter. That book was made by Mr. Mark Twain,
and he told the truth, mainly. There was things which he
stretched, but mainly he told the truth. That is nothing. I
never seen anybody but lied one time or another, without
it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly — Tom’s Aunt Polly, she is — and Mary, and the Widow
Douglas is all told about in that book, which is mostly a true
book, with some stretchers, as I said before.
Now the way that the book winds up is this: Tom and
me found the money that the robbers hid in the cave, and it
made us rich. We got six thousand dollars apiece — all gold.
It was an awful sight of money when it was piled up. Well,
Judge Thatcher he took it and put it out at interest, and it
fetched us a dollar a day apiece all the year round — more
than a body could tell what to do with. The Widow Douglas she took me for her son, and allowed she would sivilize
me; but it was rough living in the house all the time, considering how dismal regular and decent the widow was in
all her ways; and so when I couldn’t stand it no longer I lit
out. I got into my old rags and my sugar-hogshead again,
and was free and satisfied.'''
**step 1**
- text cleaning.
=> Removing stopwords and punctuation is the text cleaning.
nlp = spacy.load('en_core_web_sm')

stopwords = list(STOP_WORDS)
stopwords
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
# after this we got tokanized words(after tokazination)
# **Step 2:**
# - Word tokenization
tokens = [token.text for token in doc]
print(tokens)
punctuation = punctuation + '\n'
punctuation
# Step 4
# - word frequency table
# => here we use a condition so that we can extract important words from the text.
word_frequencies = {}
for word in doc:
  if word.text.lower() not in stopwords:
    if word.text.lower() not in punctuation:
         if word.text not in word_frequencies.keys():
            word_frequencies[word.text] = 1
         else:
          word_frequencies[word.text] += 1

print(word_frequencies)
max_frequency = max(word_frequencies.values())
max_frequency
for word in word_frequencies.keys():
   word_frequencies[word] = word_frequencies[word]/max_frequency
print(word_frequencies)
#by this we get normalized frequency of each words.
# **Step 4:**
# - sentence tokenization
sentence_tokens = [ sent for sent in doc.sents]
print(sentence_tokens)
# Now we have to calculate sentence score
sentence_scores = {}
for sent in sentence_tokens:
  for word in sent:
    if word.text.lower() in word_frequencies.keys():
      if sent not in sentence_scores.keys():
             sentence_scores[sent] = word_frequencies[word.text.lower()]
      else:
             sentence_scores[sent] += word_frequencies[word.text.lower()]
sentence_scores
from heapq import nlargest


```
# This is formatted as code
```

# Now we will get sentence with maximum score.
# and then have to select the resultant sentences with maximum frequency count.
select_length = int(len(sentence_tokens)*0.25)
select_length
summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
# **Step 5:**
# - Summary
summary
# Text summariser t5 base 
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base',return_dict = True)

sequence = ( '''YOU don’t know about me without you have read a book
by the name of The Adventures of Tom Sawyer; but that
ain’t no matter. That book was made by Mr. Mark Twain,
and he told the truth, mainly. There was things which he
stretched, but mainly he told the truth. That is nothing. I
never seen anybody but lied one time or another, without
it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly — Tom’s Aunt Polly, she is — and Mary, and the Widow
Douglas is all told about in that book, which is mostly a true
book, with some stretchers, as I said before.
Now the way that the book winds up is this: Tom and
me found the money that the robbers hid in the cave, and it
made us rich. We got six thousand dollars apiece — all gold.
It was an awful sight of money when it was piled up. Well,
Judge Thatcher he took it and put it out at interest, and it
fetched us a dollar a day apiece all the year round — more
than a body could tell what to do with. The Widow Douglas she took me for her son, and allowed she would sivilize
me; but it was rough living in the house all the time, considering how dismal regular and decent the widow was in
all her ways; and so when I couldn’t stand it no longer I lit
out. I got into my old rags and my sugar-hogshead again,
and was free and satisfied. ''' )

inputs=tokenizer.encode("sumarize: " +sequence,return_tensors='pt', max_length=812, truncation=True)
output = model.generate(inputs, min_length=40, max_length=500)
summary=tokenizer.decode(output[0])
print(summary)
# Text summariser using bert

from summarizer import Summarizer,TransformerSummarizer
text = ''' YOU don’t know about me without you have read a book
by the name of The Adventures of Tom Sawyer; but that
ain’t no matter. That book was made by Mr. Mark Twain,
and he told the truth, mainly. There was things which he
stretched, but mainly he told the truth. That is nothing. I
never seen anybody but lied one time or another, without
it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly — Tom’s Aunt Polly, she is — and Mary, and the Widow
Douglas is all told about in that book, which is mostly a true
book, with some stretchers, as I said before.
Now the way that the book winds up is this: Tom and
me found the money that the robbers hid in the cave, and it
made us rich. We got six thousand dollars apiece — all gold.
It was an awful sight of money when it was piled up. Well,
Judge Thatcher he took it and put it out at interest, and it
fetched us a dollar a day apiece all the year round — more
than a body could tell what to do with. The Widow Douglas she took me for her son, and allowed she would sivilize
me; but it was rough living in the house all the time, considering how dismal regular and decent the widow was in
all her ways; and so when I couldn’t stand it no longer I lit
out. I got into my old rags and my sugar-hogshead again,
and was free and satisfied. 
 '''
bert_model = Summarizer()
bert_summary = ''.join(bert_model(text, min_length=60))
print(bert_summary)
## **CNN DATASET FOR REST ALGORITHMS**
from datasets import load_dataset

try:
    # Load the dataset
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    print("Dataset loaded successfully.")
    
    # Example: Print the first entry from the training set
    print(dataset['train'][0])
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Features: {dataset['train'].column_names}")
sample = dataset['train'][1]
print(f'''
Article of 500 chars, total length : {len(sample['article'])}
      ''')
print(sample['article'][:500])
print(f"\n Summary (length : {len(sample['highlights'])})")
print(sample['highlights'])
sample_text = ''' YOU don’t know about me without you have read a book
by the name of The Adventures of Tom Sawyer; but that
ain’t no matter. That book was made by Mr. Mark Twain,
and he told the truth, mainly. There was things which he
stretched, but mainly he told the truth. That is nothing. I
never seen anybody but lied one time or another, without
it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly — Tom’s Aunt Polly, she is — and Mary, and the Widow
Douglas is all told about in that book, which is mostly a true
book, with some stretchers, as I said before.
Now the way that the book winds up is this: Tom and
me found the money that the robbers hid in the cave, and it
made us rich. We got six thousand dollars apiece — all gold.
It was an awful sight of money when it was piled up. Well,
Judge Thatcher he took it and put it out at interest, and it
fetched us a dollar a day apiece all the year round — more
than a body could tell what to do with. The Widow Douglas she took me for her son, and allowed she would sivilize
me; but it was rough living in the house all the time, considering how dismal regular and decent the widow was in
all her ways; and so when I couldn’t stand it no longer I lit
out. I got into my old rags and my sugar-hogshead again,
and was free and satisfied.'''
print(sample_text)
summaries = {}
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
string = "This is first sentence. This is second sentence."
sent_tokenize(string)
def three_sentence_summary(text):
  return "\n".join(sent_tokenize(text)[:3])

summaries["baseline"] = three_sentence_summary(sample_text)

print(summaries)
# GPT-2 Large
from transformers import pipeline, set_seed

set_seed(1)
# pipe = pipeline("text-generation", model="gpt2-large")
# gpt2_query = sample_text + "\nTL;DR:\n"
# pipe_out = pipe(gpt2_query,  max_length=512, clean_up_tokenization_spaces = True)
# summaries["gpt2"] = "\n".join(sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query) :]))
## **T5 Model**
pipe = pipeline("summarization", model="t5-large")
pipe_out = pipe(sample_text)
summaries["t5"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))
summaries["t5"]

print(summaries["t5"])
## **BART - facebook/bart-large-ccn**

pipe = pipeline("summarization", model="facebook/bart-large-cnn")
pipe_out = pipe(sample_text)
summaries['bart'] = "\n ".join(sent_tokenize(pipe_out[0]["summary_text"]))
summaries['bart']
print(summaries["bart"])
## **PEGASUS MODEL** 
pipe = pipeline("summarization",model= "google/pegasus-cnn_dailymail")
pipe_out = pipe(sample_text)

summaries["pegasus"] = pipe_out[0]["summary_text"]
summaries['pegasus']
print(summaries["pegasus"])
# Going over the previous fine-tuning T5 article, you will find that the T5 model can perform several tasks, including text summarization. Each task is triggered with a special token and each token corresponds to a special prepended string. For text summarization, it is the above string.
T5 comes in different sizes:

# t5-small
# t5-base
# t5-large
# t5–3b
# t5–11b.
dataset for maodels 
# Colossal Clean Crawled Corpus (C4),
# BBC newsdata set
# Wiki-DPR
# Sentence acceptability judgment
#   - CoLA [Warstadt et al., 2018](https://arxiv.org/abs/1805.12471)
# - Sentiment analysis 
#   - SST-2 [Socher et al., 2013](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
# - Paraphrasing/sentence similarity
#   - MRPC [Dolan and Brockett, 2005](https://aclanthology.org/I05-5002)
#   - STS-B [Ceret al., 2017](https://arxiv.org/abs/1708.00055)
#   - QQP [Iyer et al., 2017](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
# - Natural language inference
#   - MNLI [Williams et al., 2017](https://arxiv.org/abs/1704.05426)
#   - QNLI [Rajpurkar et al.,2016](https://arxiv.org/abs/1606.05250)
#   - RTE [Dagan et al., 2005](https://link.springer.com/chapter/10.1007/11736790_9) 
#   - CB [De Marneff et al., 2019](https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf)
# - Sentence completion
#   - COPA [Roemmele et al., 2011](https://www.researchgate.net/publication/221251392_Choice_of_Plausible_Alternatives_An_Evaluation_of_Commonsense_Causal_Reasoning)
# - Word sense disambiguation
#   - WIC [Pilehvar and Camacho-Collados, 2018](https://arxiv.org/abs/1808.09121)
# - Question answering
#   - MultiRC [Khashabi et al., 2018](https://aclanthology.org/N18-1023)
#   - ReCoRD [Zhang et al., 2018](https://arxiv.org/abs/1810.12885)
#   - BoolQ [Clark et al., 2019](https://arxiv.org/abs/1905.10044)