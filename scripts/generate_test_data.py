"""Generate test data for UI acceptance testing."""

from pathlib import Path

TEST_DATA_DIR = Path("data/raw/test_docs")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_happy_path_docs() -> None:
    """Generate documents with answerable content for happy path testing."""

    # Collection: pandas_docs (default collection)
    pandas_dir = TEST_DATA_DIR / "pandas_docs"
    pandas_dir.mkdir(exist_ok=True)

    # Document 1: DataFrame creation
    (pandas_dir / "dataframe_creation.md").write_text(
        """# Creating DataFrames

Pandas provides multiple ways to create DataFrames.

## From Dictionary

The most common way is from a dictionary:

```python
import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
df = pd.DataFrame(data)
```

## From Lists

You can also create from lists of lists:

```python
data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df = pd.DataFrame(data, columns=['name', 'age'])
```

## From CSV

Reading from CSV files:

```python
df = pd.read_csv('data.csv')
```

## Empty DataFrame

Create an empty DataFrame with specific columns:

```python
df = pd.DataFrame(columns=['name', 'age', 'city'])
```
"""
    )

    # Document 2: DataFrame operations
    (pandas_dir / "dataframe_operations.md").write_text(
        """# DataFrame Operations

Common operations on pandas DataFrames.

## Selecting Columns

Select a single column:

```python
df['column_name']
df.column_name
```

Select multiple columns:

```python
df[['col1', 'col2']]
```

## Filtering Rows

Filter rows based on conditions:

```python
df[df['age'] > 25]
df.query('age > 25')
```

## Adding Columns

Add a new column:

```python
df['new_col'] = df['col1'] + df['col2']
```

## Grouping

Group by column and aggregate:

```python
df.groupby('category')['value'].sum()
df.groupby('category').agg({'value': 'sum', 'count': 'count'})
```
"""
    )

    # Document 3: Indexing
    (pandas_dir / "indexing.md").write_text(
        """# Indexing and Selection

How to select data from DataFrames.

## loc and iloc

Use `loc` for label-based indexing:

```python
df.loc[0, 'column_name']
df.loc[0:5, ['col1', 'col2']]
```

Use `iloc` for integer-based indexing:

```python
df.iloc[0, 0]
df.iloc[0:5, 0:3]
```

## Boolean Indexing

Filter using boolean arrays:

```python
df[df['age'] > 30]
df[(df['age'] > 30) & (df['city'] == 'NYC')]
```

## Setting Values

Modify values using loc:

```python
df.loc[0, 'age'] = 26
df.loc[df['age'] > 30, 'status'] = 'senior'
```
"""
    )

    num_docs = len(list(pandas_dir.glob("*.md")))
    print(f"[PASS] Generated {num_docs} happy path documents in {pandas_dir}")


def generate_refusal_path_docs() -> None:
    """Generate documents with unrelated content for refusal testing."""

    # Collection: unrelated_docs (for refusal testing)
    unrelated_dir = TEST_DATA_DIR / "unrelated_docs"
    unrelated_dir.mkdir(exist_ok=True)

    (unrelated_dir / "cooking.md").write_text(
        """# Cooking Recipes

This document is about cooking, not programming.

## Italian Pasta

To make pasta, boil water and add salt. Cook pasta for 8-12 minutes.

## Baking Bread

Mix flour, water, yeast, and salt. Knead for 10 minutes. Let rise for 1 hour.
"""
    )

    (unrelated_dir / "travel.md").write_text(
        """# Travel Guide

Information about travel destinations.

## Paris

Paris is the capital of France. Visit the Eiffel Tower and Louvre Museum.

## Tokyo

Tokyo is the capital of Japan. Famous for sushi and cherry blossoms.
"""
    )

    (unrelated_dir / "sports.md").write_text(
        """# Sports Information

Various sports and their rules.

## Soccer

Soccer is played with 11 players per team. The field is 100-110 meters long.

## Basketball

Basketball is played with 5 players per team. The court is 28 meters long.
"""
    )

    num_docs = len(list(unrelated_dir.glob("*.md")))
    print(f"[PASS] Generated {num_docs} refusal test documents in {unrelated_dir}")


def generate_alternate_collection_docs() -> None:
    """Generate documents for testing collection switching."""

    # Collection: numpy_docs (alternate collection)
    numpy_dir = TEST_DATA_DIR / "numpy_docs"
    numpy_dir.mkdir(exist_ok=True)

    (numpy_dir / "arrays.md").write_text(
        """# NumPy Arrays

Creating and working with NumPy arrays.

## Creating Arrays

From Python lists:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
```

From zeros:

```python
arr = np.zeros((3, 4))
```

From ones:

```python
arr = np.ones((2, 3))
```

## Array Operations

Element-wise operations:

```python
arr1 + arr2
arr1 * arr2
np.sqrt(arr)
```
"""
    )

    (numpy_dir / "indexing.md").write_text(
        """# NumPy Array Indexing

How to access elements in NumPy arrays.

## Basic Indexing

```python
arr[0]
arr[0, 1]
arr[0:5]
```

## Boolean Indexing

```python
arr[arr > 5]
arr[(arr > 5) & (arr < 10)]
```

## Fancy Indexing

```python
arr[[0, 2, 4]]
arr[[0, 1], [0, 1]]
```
"""
    )

    num_docs = len(list(numpy_dir.glob("*.md")))
    print(f"[PASS] Generated {num_docs} alternate collection documents in {numpy_dir}")


def generate_test_queries_file() -> None:
    """Generate a file with test queries for each scenario."""

    queries_content = """# Test Queries for UI Acceptance Testing

## A. Happy Path Tests (should return answers + sources)

### Collection: pandas_docs

1. "How do I create a DataFrame from a dictionary?"
   - Expected: Answer about pd.DataFrame(dict) with sources

2. "What are the different ways to create a DataFrame?"
   - Expected: Multiple methods (dict, lists, CSV, empty) with sources

3. "How do I filter rows in a DataFrame?"
   - Expected: Answer about boolean indexing with sources

4. "Explain how to use loc and iloc for indexing"
   - Expected: Answer about label-based vs integer-based indexing

5. "How do I group data in pandas?"
   - Expected: Answer about groupby() with sources

## B. Refusal Path Tests (should return refusal message, NO sources)

### Collection: pandas_docs (but ask unrelated questions)

1. "How do I make pasta?"
   - Expected: Refusal message, no sources expander

2. "What is the capital of France?"
   - Expected: Refusal message, no sources expander

3. "asdf qwer zxcv"
   - Expected: Refusal message, no sources expander

4. "Tell me about soccer rules"
   - Expected: Refusal message, no sources expander

## C. Debug Mode Tests

Use any happy path query with Debug Mode enabled:
- Expected: Debug expander shows context_used

## D. Reranker Toggle Tests

Use same query with reranker ON vs OFF:
- Expected: Different ranking (or at least no crash)

## E. Collection Switch Tests

1. Switch to "numpy_docs" collection
   - Query: "How do I create a NumPy array?"
   - Expected: Answer about np.array() with sources

2. Switch back to "pandas_docs"
   - Query: "How do I create a DataFrame?"
   - Expected: Answer about pandas DataFrame

## F. Error Path Tests

1. Stop Ollama service
2. Ask any question
3. Expected: Clean OllamaConnectionError message displayed
"""

    queries_file = TEST_DATA_DIR / "TEST_QUERIES.md"
    queries_file.write_text(queries_content)
    print(f"[PASS] Generated test queries file: {queries_file}")


def main() -> None:
    """Generate all test data."""
    print("Generating test data for UI acceptance testing...\n")

    generate_happy_path_docs()
    generate_refusal_path_docs()
    generate_alternate_collection_docs()
    generate_test_queries_file()

    print("\n[SUCCESS] Test data generation complete!")
    print("\nNext steps:")
    print("1. Ingest pandas_docs collection:")
    print(
        "   python -m src.cli ingest --input data/raw/test_docs/pandas_docs "
        "--collection pandas_docs --library pandas --version 2.0.0"
    )
    print("\n2. Ingest numpy_docs collection:")
    print(
        "   python -m src.cli ingest --input data/raw/test_docs/numpy_docs "
        "--collection numpy_docs --library numpy --version 1.24.0"
    )
    print("\n3. Ingest unrelated_docs collection (for refusal testing):")
    print(
        "   python -m src.cli ingest --input data/raw/test_docs/unrelated_docs "
        "--collection unrelated_docs"
    )
    print("\n4. Use queries from data/raw/test_docs/TEST_QUERIES.md for testing")


if __name__ == "__main__":
    main()
