from starter_code import *
from collections import Counter, defaultdict
import math
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

'''
A class that represents the Hidden Markov Model for Part-of-Speech tagging.

Arguments:
    - tagged_sents: a list of sentences, where each sentence is a list of tuples (word, tag)
    - alpha: the smoothing parameter for the model (default is 1.0)

Methods:
    - get_tags(): calculates the unigram and bigram tag counts, as well as the vocabulary size
    - get_transition_probs(): calculates the transition probabilities between tags
    - get_emission_probs(): calculates the emission probabilities from tags to words
    - _compute_log_probs(): precomputes the log probabilities for faster decoding ()
'''
class POS_HMM:
    def __init__(self, tagged_sents, alpha=1.0):
        self.tagged_sents = tagged_sents
        self.alpha = alpha
        
        self.tags, self.num_tags, self.vocab_size, self.unigram_tag_counts, self.transition_counts, self.emission_counts = self.get_tags()
        self.transition_probs = self.get_transition_probs()
        self.emission_probs = self.get_emission_probs()
        self.sentence_count = len(self.tagged_sents)

        self.log_transition_probs = self._compute_log_probs(self.transition_probs)
        self.log_emission_probs = self._compute_log_probs(self.emission_probs)
        self.default_transition_log_prob = math.log(alpha / self.num_tags)
        self.default_emission_log_prob = math.log(alpha / self.vocab_size)

    # First, we need to get all of the tags, their counts, and the number of unique tags
    def get_tags(self):
        # Unigram tag counts
        unigram_tag_counts = Counter()

        # Bigram tag counts; nested dictionary
        transition_counts = defaultdict(Counter)

        vocabulary = set()

        # Emission counts (tag, word pairs); nested dictionary
        emission_counts = defaultdict(Counter)

        for sent in self.tagged_sents:
            # add start and stop tokens
            sent.insert(0, ('<START>', ''))
            sent.append(('<STOP>', ''))

            prev_tag = None

            for word, tag in sent:
                unigram_tag_counts[tag] += 1
                emission_counts[tag][word] += 1
                vocabulary.add(word)

                if prev_tag:
                    transition_counts[prev_tag][tag] += 1

                prev_tag = tag

        tags = list(unigram_tag_counts.keys())
        num_tags = len(tags)
        vocab_size = len(vocabulary)

        return tags, num_tags, vocab_size, unigram_tag_counts, transition_counts, emission_counts


    def get_transition_probs(self):
        transition_probs = defaultdict(dict)
    
        
        # For each tag, get all transitions and their counts
        for tag1 in self.transition_counts:

            # get the total count of the previous tag
            total_prev_tag_count = self.unigram_tag_counts[tag1]

            # get all of the transitions from the previous tag to current tag
            for tag2, count in self.transition_counts[tag1].items():

                try:
                    bigram_count = self.transition_counts[tag1][tag2]

                # if the transition is not found, set the count to 0
                except KeyError:
                    bigram_count = 0

                transition_prob = (bigram_count + self.alpha) / (total_prev_tag_count + self.alpha * self.num_tags)
                transition_probs[tag1][tag2] = transition_prob

        return transition_probs

    def get_emission_probs(self):
    # find the probability of a word given a tag

        emission_probs = defaultdict(dict)

        for tag in self.emission_counts:
            # total count of the tag
            total_tag_count = self.unigram_tag_counts[tag]

            # get the count of each word with the tag
            for word in self.emission_counts[tag]:
                emission_prob = (self.emission_counts[tag][word] + self.alpha) / (total_tag_count + self.alpha * self.vocab_size)
                emission_probs[tag][word] = emission_prob

        return emission_probs
    
    # precompute log probs to speed up decoding
    def _compute_log_probs(self, prob_dict):
        return {
            outer_key: {
                inner_key: math.log(prob)
                for inner_key, prob in inner_dict.items()
            }
            for outer_key, inner_dict in prob_dict.items()
        }

'''
A class that represents the Viterbi algorithm for decoding the POS tags.

Arguments:
    - pos_hmm: an instance of the POS_HMM class

Methods:
    - decode(): decodes the sentence using the Viterbi algorithm
    - get_sequence_score(): calculates the score of a sequence of tags for a given sentence
'''
class Viterbi:
    def __init__(self, pos_hmm):
        self.pos_hmm = pos_hmm

    def decode(self, sentence):
        len_sentence = len(sentence)
        tags = self.pos_hmm.tags
        
        # pre-compute emission probabilities before the algorithm
        emission_probs = [{
            tag: self.pos_hmm.log_emission_probs[tag].get(word, self.pos_hmm.default_emission_log_prob)
            for tag in tags
        } for word in sentence]

        # set up dynamic programming and backpointer arrays
        dp = [{} for _ in range(len_sentence)]
        backpointer = [{} for _ in range(len_sentence)]

        # initialize the algorithm at the first word
        dp[0] = {tag: emission_probs[0][tag] for tag in tags}
        backpointer[0] = {tag: None for tag in tags}
        
        # Viterbi algorithm starts 
        for t in range(1, len_sentence):
            for curr_tag in tags:
                curr_emission = emission_probs[t][curr_tag]
                max_prob, best_prev_tag = max(
                    (dp[t-1][prev_tag] + 
                     self.pos_hmm.log_transition_probs.get(prev_tag, {}).get(curr_tag, self.pos_hmm.default_transition_log_prob) +
                     curr_emission,
                     prev_tag)
                    for prev_tag in tags
                )
                dp[t][curr_tag] = max_prob
                backpointer[t][curr_tag] = best_prev_tag

        # find the best end tag
        best_end_tag = max(dp[len_sentence-1], key=dp[len_sentence-1].get)
        best_path = [best_end_tag]
        
        # backtrack to find the best path, store the tags in reverse order
        for t in range(len_sentence - 1, 0, -1):
            best_path.append(backpointer[t][best_path[-1]])

        best_path.reverse()
        return best_path


    # calculate the score of a sequence of tags for a given sentence -- used for debugging
    # should have probably used the evaluate() method provided by the instructor
    def get_sequence_score(self, sentence, tag_sequence):
        log_score = 0.0 
        for i in range(len(sentence)):

            # emission probability
            tag = tag_sequence[i]
            word = sentence[i]
            emission_prob = self.pos_hmm.emission_probs[tag].get(word, self.pos_hmm.alpha / len(self.pos_hmm.tags))
            log_score += math.log(emission_prob)
            
            # transition probability (we don't need it for first tag)
            if i > 0:
                prev_tag = tag_sequence[i-1]
                transition_prob = self.pos_hmm.transition_probs[prev_tag].get(tag, self.pos_hmm.alpha / len(self.pos_hmm.tags))
                log_score += math.log(transition_prob)
            
        return log_score

'''
A function that tunes the alpha parameter for the HMM.

Arguments:
    - train_sentences: a list of sentences, where each sentence is a list of tuples (word, tag)
    - dev_sentences: a list of sentences, where each sentence is a list of tuples (word, tag)
    - alpha_values: a list of alpha values to test

Returns:
    - best_alpha: the alpha value that achieves the highest accuracy
    - results: a dictionary of alpha values and their corresponding accuracies -- assumes we want to maximize the micro-average F1 score 
'''
def tune_alpha(train_sentences, dev_sentences, alpha_values):
    best_accuracy = 0
    best_alpha = None
    results = {}

    for alpha in alpha_values:
        print(f"Testing alpha = {alpha}")

        # train model with current alpha
        pos_hmm = POS_HMM(tagged_sents=train_sentences, alpha=alpha)
        viterbi = Viterbi(pos_hmm)
        
        # evaluate on dev set
        dev_tagged_sentences = []
        for sentence in dev_sentences:
            words = [word for word, tag in sentence]
            predicted_tags = viterbi.decode(words)
            tagged_sentence = list(zip(words, predicted_tags))
            dev_tagged_sentences.append(tagged_sentence)
        
        # accuracy for current alpha
        accuracy = evaluate(dev_sentences, dev_tagged_sentences, print_results=False) # I added a print_results argument to the evaluate() fun
        results[alpha] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
    
    return best_alpha, results


# sklearn's confusion matrix of the entire dataset was too large and difficult to visualize, 
# so I used these two functions to generate and print a confusion matrix of the top n tags
def compute_confusion_matrix(true_sentences, pred_sentences, tags):
    confusion_matrix = defaultdict(Counter)
    
    for true_sent, pred_sent in zip(true_sentences, pred_sentences):
        for (_, true_tag), (_, pred_tag) in zip(true_sent, pred_sent):
            confusion_matrix[true_tag][pred_tag] += 1
    
    return confusion_matrix

def print_confusion_matrix(confusion_matrix, tags, top_n=10):
    tag_counts = Counter()
    for true_tag, pred_counts in confusion_matrix.items():
        tag_counts[true_tag] = sum(pred_counts.values())
    
    most_common_tags = [tag for tag, _ in tag_counts.most_common(top_n)]
    
    # create a matrix for sklearn
    matrix = np.zeros((top_n, top_n))
    for i, true_tag in enumerate(most_common_tags):
        for j, pred_tag in enumerate(most_common_tags):
            matrix[i][j] = confusion_matrix[true_tag][pred_tag]
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=most_common_tags
    )
    
    plt.figure(figsize=(10, 8))
    disp.plot(xticks_rotation=45)
    plt.title(f'Confusion Matrix (top {top_n} tags)')
    plt.tight_layout()
    plt.show()

def main():
    datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")
    train, dev, test = load_treebank_splits(datadir)

    train_sentences = [get_token_tag_tuples(sent) for sent in train]
    dev_sentences = [get_token_tag_tuples(sent) for sent in dev]
    
    # # Tune alpha using the development set
    # alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    # start_time = time.time()
    # best_alpha, tuning_results = tune_alpha(train_sentences, dev_sentences, alpha_values)
    # end_time = time.time()
    # print(f"Time taken for tuning: {end_time - start_time:.2f} seconds")
    
    # print("\nTuning Results:")
    # for alpha, accuracy in tuning_results.items():
    #     print(f"Alpha: {alpha}, Accuracy: {accuracy:.4f}")
    # print(f"\nBest alpha: {best_alpha}")

    best_alpha = 0.001  # this was the best alpha found by the tuning process
    
    # train final model with best alpha
    pos_hmm = POS_HMM(tagged_sents=train_sentences, alpha=best_alpha)
    viterbi = Viterbi(pos_hmm)
    
    # evaluate on test set
    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    test_tagged_sentences = []
    
    for sentence in test_sentences:
        words = [word for word, tag in sentence]
        predicted_tags = viterbi.decode(words)
        tagged_sentence = list(zip(words, predicted_tags))
        test_tagged_sentences.append(tagged_sentence)
    
    print("\nFinal Test Set Results (with best alpha):")
    evaluate(test_sentences, test_tagged_sentences)
    
    # # Finding and printing misclassifications of NN->NNP
    # print("\nExamples of NN->NNP misclassifications:")
    # for true_sent, pred_sent in zip(test_sentences, test_tagged_sentences):
    #     for (word, true_tag), (_, pred_tag) in zip(true_sent, pred_sent):
    #         if true_tag == 'NN' and pred_tag == 'NNP':
    #             # Print the full sentence with the misclassified word highlighted
    #             print("\nSentence:")
    #             for (w, t), (_, p) in zip(true_sent, pred_sent):
    #                 if w == word and t == 'NN' and p == 'NNP':
    #                     print(f"{w}[True:NN, Pred:NNP]", end=" ")
    #                 else:
    #                     print(w, end=" ")
    #             print("\n")
    #             break
    
    # Compute and display confusion matrix
    # confusion_matrix = compute_confusion_matrix(test_sentences, test_tagged_sentences, pos_hmm.tags)
    # print_confusion_matrix(confusion_matrix, pos_hmm.tags)

if __name__ == "__main__":
    main()