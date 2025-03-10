"""
Text Transformers Module

This module provides transformers for text features in the NexusML suite.
Each transformer follows the Single Responsibility Principle (SRP) from SOLID,
focusing on a specific text transformation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd

from nexusml.core.feature_engineering.base import BaseColumnTransformer, BaseConfigurableTransformer
from nexusml.core.feature_engineering.registry import register_transformer


class TextCombiner(BaseColumnTransformer):
    """
    Combines multiple text columns into one column.
    
    This transformer takes multiple text columns and combines them into a single
    text column using a specified separator.
    
    Config example: {"columns": ["Asset Category","Equip Name ID"], "separator": " "}
    """
    
    def __init__(
        self,
        columns: List[str],
        separator: str = " ",
        new_column: str = "combined_text",
        name: str = "TextCombiner",
    ):
        """
        Initialize the text combiner.
        
        Args:
            columns: Names of the columns to combine.
            separator: Separator to use between column values.
            new_column: Name of the new column to create.
            name: Name of the transformer.
        """
        super().__init__(columns, [new_column], name)
        self.separator = separator
        self.new_column = new_column
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Combine the specified columns into a single text column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the combined text column.
        """
        # Create a single text column from all specified columns
        X[self.new_column] = (
            X[self.input_columns]
            .astype(str)
            .apply(lambda row: self.separator.join(row.values), axis=1)
        )
        
        # Fill NaN values with empty string
        X[self.new_column] = X[self.new_column].fillna("")
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If some columns are missing, use only the available ones.
        If all columns are missing, create an empty column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Find available columns
        available_columns = [col for col in self.input_columns if col in X.columns]
        
        if not available_columns:
            # If no columns are available, create an empty column
            X[self.new_column] = ""
            return X
        
        # Create a single text column from available columns
        X[self.new_column] = (
            X[available_columns]
            .astype(str)
            .apply(lambda row: self.separator.join(row.values), axis=1)
        )
        
        # Fill NaN values with empty string
        X[self.new_column] = X[self.new_column].fillna("")
        
        return X


class TextNormalizer(BaseColumnTransformer):
    """
    Normalizes text in a column.
    
    This transformer applies various normalization techniques to text data,
    such as lowercasing, removing special characters, and stemming.
    
    Config example: {
        "column": "description",
        "new_column": "normalized_description",
        "lowercase": true,
        "remove_special_chars": true,
        "remove_stopwords": true,
        "stemming": false
    }
    """
    
    def __init__(
        self,
        column: str,
        new_column: Optional[str] = None,
        lowercase: bool = True,
        remove_special_chars: bool = False,
        remove_stopwords: bool = False,
        stemming: bool = False,
        name: str = "TextNormalizer",
    ):
        """
        Initialize the text normalizer.
        
        Args:
            column: Name of the column to normalize.
            new_column: Name of the new column to create. If None, overwrites the input column.
            lowercase: Whether to convert text to lowercase.
            remove_special_chars: Whether to remove special characters.
            remove_stopwords: Whether to remove stopwords.
            stemming: Whether to apply stemming.
            name: Name of the transformer.
        """
        output_column = new_column or column
        super().__init__([column], [output_column], name)
        self.column = column
        self.new_column = output_column
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        
        # Initialize NLP components if needed
        if remove_stopwords or stemming:
            try:
                import nltk
                
                # Download required NLTK resources
                nltk.download("stopwords", quiet=True)
                nltk.download("punkt", quiet=True)
                
                if stemming:
                    from nltk.stem import PorterStemmer
                    self.stemmer = PorterStemmer()
                
                if remove_stopwords:
                    from nltk.corpus import stopwords
                    self.stopwords = set(stopwords.words("english"))
            except ImportError:
                raise ImportError("NLTK is required for stopword removal and stemming. Install it with 'pip install nltk'.")
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the text in the specified column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the normalized text column.
        """
        # Create a copy of the column to normalize
        X[self.new_column] = X[self.column].astype(str)
        
        # Apply normalization techniques
        if self.lowercase:
            X[self.new_column] = X[self.new_column].str.lower()
        
        if self.remove_special_chars:
            import re
            X[self.new_column] = X[self.new_column].apply(
                lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x)
            )
        
        if self.remove_stopwords:
            X[self.new_column] = X[self.new_column].apply(self._remove_stopwords)
        
        if self.stemming:
            X[self.new_column] = X[self.new_column].apply(self._apply_stemming)
        
        return X
    
    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the text.
        
        Args:
            text: Text to process.
            
        Returns:
            Text with stopwords removed.
        """
        if not hasattr(self, "stopwords"):
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return " ".join(filtered_words)
    
    def _apply_stemming(self, text: str) -> str:
        """
        Apply stemming to the text.
        
        Args:
            text: Text to process.
            
        Returns:
            Stemmed text.
        """
        if not hasattr(self, "stemmer"):
            return text
        
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, create an empty output column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create an empty output column
        X[self.new_column] = ""
        return X


class TextTokenizer(BaseColumnTransformer):
    """
    Tokenizes text in a column.
    
    This transformer splits text into tokens and creates a new column with the tokens.
    
    Config example: {
        "column": "description",
        "new_column": "tokens",
        "lowercase": true,
        "min_token_length": 2,
        "max_tokens": 100
    }
    """
    
    def __init__(
        self,
        column: str,
        new_column: Optional[str] = None,
        lowercase: bool = True,
        min_token_length: int = 2,
        max_tokens: Optional[int] = None,
        name: str = "TextTokenizer",
    ):
        """
        Initialize the text tokenizer.
        
        Args:
            column: Name of the column to tokenize.
            new_column: Name of the new column to create. If None, uses "{column}_tokens".
            lowercase: Whether to convert text to lowercase before tokenizing.
            min_token_length: Minimum length of tokens to keep.
            max_tokens: Maximum number of tokens to keep. If None, keeps all tokens.
            name: Name of the transformer.
        """
        output_column = new_column or f"{column}_tokens"
        super().__init__([column], [output_column], name)
        self.column = column
        self.new_column = output_column
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.max_tokens = max_tokens
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenize the text in the specified column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the tokenized text column.
        """
        # Create a copy of the column to tokenize
        text_series = X[self.column].astype(str)
        
        # Apply lowercase if needed
        if self.lowercase:
            text_series = text_series.str.lower()
        
        # Tokenize the text
        import re
        
        def tokenize(text: str) -> List[str]:
            # Split text into tokens
            tokens = re.findall(r"\b\w+\b", text)
            
            # Filter tokens by length
            tokens = [token for token in tokens if len(token) >= self.min_token_length]
            
            # Limit the number of tokens if needed
            if self.max_tokens is not None:
                tokens = tokens[:self.max_tokens]
            
            return tokens
        
        # Apply tokenization
        X[self.new_column] = text_series.apply(tokenize)
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, create an empty output column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create an empty output column
        X[self.new_column] = X.apply(lambda _: [], axis=1)
        return X


# Register transformers with the registry
register_transformer("text_combiner", TextCombiner)
register_transformer("text_normalizer", TextNormalizer)
register_transformer("text_tokenizer", TextTokenizer)