import nltk

# Your original summary
summary = "YOU don’t know about me without you have read a book by the name of The Adventures of Tom Sawyer but that ain’t no matter. That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth."

# Tokenize the summary into sentences
sentences = nltk.sent_tokenize(summary)

# Determine the number of sentences per segment
sentences_per_segment = len(sentences) // 3

# Generate candidates by dividing the sentences into segments
references = []
for i in range(0, len(sentences), sentences_per_segment):
    segment = " ".join(sentences[i:i+sentences_per_segment])
    references.append(segment.strip())

print(references)

from evaluate import load
# Load the ROUGE metric
import evaluate
rouge = evaluate.load('rouge')
candidates = ["YOU don't know me","read the book the Adventures of Tom Sawyer","he told the truth"]

references = ["YOU don’t know about me without you have read a book by the name of The Adventures of Tom Sawyer but that ain’t no matter.", 
"That book was made by Mr. Mark Twain, and he told the truth, mainly.", "There was things which he stretched, but mainly he told the truth."]
results = rouge.compute(predictions=candidates, references=references)
print(results)