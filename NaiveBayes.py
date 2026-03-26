import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self) -> None:
        """Initializes the Naive Bayes classifier with empty probability tables.

        Attributes:
            log_prior_ham (float): Log probability of a message being ham.
            log_prior_spam (float): Log probability of a message being spam.
            ham_word_counts (defaultdict): Word counts across all ham messages.
            spam_word_counts (defaultdict): Word counts across all spam messages.
            ham_word_proba (dict): Log probability of each word given ham.
            spam_word_proba (dict): Log probability of each word given spam.
            vocabulary (set): All unique words seen during training.
        """
        self.log_prior_ham  = 0.0
        self.log_prior_spam = 0.0

        self.ham_word_counts  = defaultdict(int)
        self.spam_word_counts = defaultdict(int)

        self.ham_word_probability  = {}
        self.spam_word_probability = {}

        self.vocabulary = set()

    def train(self, x_train: list[list[str]], y_train: list[int]) -> None:
        """Trains the Naive Bayes classifier using the training data.

        Args:
            x_train (list[list[str]]): List of token lists (preprocessed messages).
            y_train (list[int]): List of integer labels (1 = spam, 0 = ham).
        """
        n = len(y_train)

        # Compute class priors: log P(ham) and log P(spam)
        ham_lines  = y_train.count(0)
        spam_lines = y_train.count(1)

        self.log_prior_ham  = math.log(ham_lines  / n)
        self.log_prior_spam = math.log(spam_lines / n)

        # Count word occurrences separately for ham and spam
        for words, label in zip(x_train, y_train):
            for word in words:
                self.vocabulary.add(word)
                if label == 0:
                    self.ham_word_counts[word]  += 1
                else:
                    self.spam_word_counts[word] += 1

        vocab_size       = len(self.vocabulary)
        total_ham_words  = sum(self.ham_word_counts.values())
        total_spam_words = sum(self.spam_word_counts.values())

        # Compute log P(word | class) with Laplace smoothing
        # P(word | ham)  = (count(word in ham)  + 1) / (total ham words  + vocab size)
        # P(word | spam) = (count(word in spam) + 1) / (total spam words + vocab size)
        for word in self.vocabulary:
            self.ham_word_probability[word]  = math.log((self.ham_word_counts[word]  + 1) / (total_ham_words  + vocab_size))
            self.spam_word_probability[word] = math.log((self.spam_word_counts[word] + 1) / (total_spam_words + vocab_size))

    def prediction(self, x_test: list[list[str]]) -> list[int]:
        """Predicts the probability of each message being spam or ham.

        Args:
            x_test (list[list[str]]): List of token lists (preprocessed messages).

        Returns:
            list[int]: Predicted labels for each message (1 = spam, 0 = ham).
        """
        predictions = []

        for words in x_test:
            # Start from the log prior for each class
            ham_score  = self.log_prior_ham
            spam_score = self.log_prior_spam

            # Add log probability of each word given the class
            for word in words:
                if word in self.ham_word_probability and word in self.spam_word_probability:
                    ham_score  += self.ham_word_probability[word]
                    spam_score += self.spam_word_probability[word]

            # Assign the label with the higher score
            predictions.append(0 if ham_score > spam_score else 1)

        return predictions