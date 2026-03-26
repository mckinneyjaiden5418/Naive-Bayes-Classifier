from nltk.stem import PorterStemmer
import re
import random

class DataLoader:
    """Handles data preprocessing, loading, and splitting."""
    @staticmethod
    def preprocess(text: str) -> list[str]:
        """Converts text to lowercase, removes special characters, tokenizes, and applies stemming.

        Args:
            text (str): Raw message string.

        Returns:
            list[str]: Cleaned and stemmed word tokens.
        """
        # Convert text to lower case
        text = text.lower()
        
        # Remove special charactors using re
        text = re.sub(r'[^a-z0-9\s]', '', text)

        # Split text into [words]
        words = text.split()

        # Apply stemming to each word using PorterStemmer
        stemmer = PorterStemmer()

        # Return cleaned text
        cleaned_words = [stemmer.stem(word) for word in words]
        return cleaned_words

    @staticmethod
    def load_data(file_path: str) -> tuple[list[list[str]], list[int]]:
        """Loads and preprocesses data from a file, returning features and labels.

        Args:
            file_path (str): Path to the tab-separated dataset file.

        Returns:
            tuple: (x, y) where x is a list of token lists and y is a list
                of integer labels (1 = spam, 0 = ham).
        """
        x, y = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label, text = line.split('\t', 1)
                y.append(1 if label == 'spam' else 0)
                x.append(DataLoader.preprocess(text))
        return x, y

    @staticmethod
    def split_data(x: list[list[str]], y: list[int], test_ratio=0.2) -> tuple[list[list[str]], list[list[str]], list[int], list[int]]:
        """Splits data into training and testing sets manually.
        Args:
            x (list): List of token lists (preprocessed messages).
            y (list): List of integer labels (1 = spam, 0 = ham).
            test_ratio (float): Fraction of data to reserve for testing (default 0.2).

        Returns:
            tuple: (x_train, x_test, y_train, y_test) split according to test_ratio.
        """
        # Create a list of positions [0, 1, 2, ...] and shuffle them
        positions = list(range(len(x)))
        random.shuffle(positions)

        # Find where to cut — everything before is train, after is test
        cutoff = int(len(positions) * (1 - test_ratio))

        # Split positions into two groups
        train_positions = positions[:cutoff]
        test_positions  = positions[cutoff:]

        # Use those positions to grab the matching messages and labels
        x_train = [x[i] for i in train_positions]
        y_train = [y[i] for i in train_positions]
        x_test  = [x[i] for i in test_positions]
        y_test  = [y[i] for i in test_positions]

        return x_train, x_test, y_train, y_test