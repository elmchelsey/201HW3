from starter_code import *
from collections import Counter, defaultdict
import math
import time

class POS_HMM:
    def __init__(self, tagged_sents, alpha=1.0):
        self.tagged_sents = tagged_sents
        self.alpha = alpha
        
        self.tags, self.num_tags, self.vocab_size, self.unigram_tag_counts, self.transition_counts, self.emission_counts = self.get_tags()
        self.transition_probs = self.get_transition_probs()
        self.emission_probs = self.get_emission_probs()
        self.sentence_count = len(self.tagged_sents)

        # Add log probability calculations during initialization
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
    
    def _compute_log_probs(self, prob_dict):
        return {
            outer_key: {
                inner_key: math.log(prob)
                for inner_key, prob in inner_dict.items()
            }
            for outer_key, inner_dict in prob_dict.items()
        }

class Viterbi:
    def __init__(self, pos_hmm):
        self.pos_hmm = pos_hmm

    def decode(self, sentence):
        len_sentence = len(sentence)
        tags = self.pos_hmm.tags
        
        # Pre-compute emission probabilities for the sentence
        emission_probs = [{
            tag: self.pos_hmm.log_emission_probs[tag].get(word, self.pos_hmm.default_emission_log_prob)
            for tag in tags
        } for word in sentence]

        dp = [{} for _ in range(len_sentence)]
        backpointer = [{} for _ in range(len_sentence)]

        # Initialize first step
        dp[0] = {tag: emission_probs[0][tag] for tag in tags}
        backpointer[0] = {tag: None for tag in tags}
        
        # Main Viterbi algorithm
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

        best_end_tag = max(dp[len_sentence-1], key=dp[len_sentence-1].get)
        best_path = [best_end_tag]
        
        for t in range(len_sentence - 1, 0, -1):
            best_path.append(backpointer[t][best_path[-1]])

        best_path.reverse()
        return best_path

    def get_sequence_score(self, sentence, tag_sequence):
        log_score = 0.0  # Start with 0 since we're adding logs
        for i in range(len(sentence)):
            # Emission probability
            tag = tag_sequence[i]
            word = sentence[i]
            emission_prob = self.pos_hmm.emission_probs[tag].get(word, self.pos_hmm.alpha / len(self.pos_hmm.tags))
            log_score += math.log(emission_prob)
            
            # Transition probability (skip for first tag)
            if i > 0:
                prev_tag = tag_sequence[i-1]
                transition_prob = self.pos_hmm.transition_probs[prev_tag].get(tag, self.pos_hmm.alpha / len(self.pos_hmm.tags))
                log_score += math.log(transition_prob)
            
        return log_score

def main():
    datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")
    train, dev, test = load_treebank_splits(datadir)

    train_sentences = [get_token_tag_tuples(sent) for sent in train]
    
    # Train and decode the HMM
    start_time = time.time()
    pos_hmm = POS_HMM(tagged_sents=train_sentences)
    viterbi = Viterbi(pos_hmm)
    end_time = time.time()
    print(f"Time taken to train and decode HMM: {end_time - start_time} seconds")
    
    dev_sentences = [get_token_tag_tuples(sent) for sent in dev]
    dev_predictions = []
    dev_gold = []
    
    batch_size = 100
    for i in range(0, len(dev_sentences), batch_size):
        batch = dev_sentences[i:i + batch_size]
        words_batch = [[word for word, tag in sent] for sent in batch]
        tags_batch = [[tag for word, tag in sent] for sent in batch]
        
        # Process each sentence in the batch
        for words in words_batch:
            dev_predictions.extend(viterbi.decode(words))
        dev_gold.extend([tag for sent_tags in tags_batch for tag in sent_tags])
    
    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    test_predictions = []
    test_gold = []
    
    for sentence in test_sentences:
        words = [word for word, tag in sentence]
        predicted_tags = viterbi.decode(words)
        test_predictions.extend(predicted_tags)
        test_gold.extend([tag for word, tag in sentence])
    
    print("Dev set accuracy:", evaluate(dev_predictions, dev_gold))
    print("Test set accuracy:", evaluate(test_predictions, test_gold))

if __name__ == "__main__":
    main()