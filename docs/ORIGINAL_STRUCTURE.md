# Original Repository Structure Analysis

This document captures the original architecture and design of the arXiv literature corpus creation system before refactoring.

## Overview

This repository contains a complete pipeline for **creating a literature corpus focused on agent-related topics from arXiv papers**. The project builds a comprehensive dataset of scientific articles on the concept of "agents" in computer science and AI, with emphasis on different agent types (autonomous agents, LLM-based agents, reinforcement learning agents, AGI, simulation-based agents) and related components (action, memory, reward, state, policy, environment, etc.).

---

## Overall Architecture & Workflow

The project follows a **multi-stage pipeline** with 6 sequential steps:

```
STEP 1: Query arXiv API → Generate search results
        ↓
STEP 2: Sort search results by URL
        ↓
STEP 3: Remove duplicates & count occurrences
        ↓
STEP 4: Re-sort by duplicate count
        ↓
STEP 5: Create Excel from text files (aggregate results)
        ↓
STEP 6: Cross-reference occurrences across search queries
        ↓
HitPyRank Processing (2 modes) → Final corpus with paragraphs
        ↓
Manual Selection & Annotation → Final refined corpus
```

---

## Directory Structure

```
Creating-a-literature-corpus-based-on-Arxiv/
├── README.md
│
├── Codes extracting the corpus/
│   ├── STEP 1: Downloading results search queries from ArXiv  (no extension)
│   ├── STEP 2: Sorting the results of the search queries      (no extension)
│   ├── STEP 3: Removing and counting duplicates               (no extension)
│   ├── STEP 4: Sorting Output step 3                          (no extension)
│   ├── STEP 5 Finding most common results in search queries, step 1  (no extension)
│   └── STEP 6 Finding most common results in search queries   (no extension)
│
├── Codes for extracting selections/
│   └── Extract colored rows                                    (no extension)
│
├── Files used in extracting the corpus/
│   ├── STEP 1: Input - Complete list of search combinations
│   └── STEP 1: Results search queries/
│       └── [50 .txt files with arXiv paper URLs]
│
├── HitPyRank/
│   ├── Mode A/
│   │   ├── HitPyRank.v1.py                    (469 lines - main NLP tool)
│   │   ├── agent - Blad1.tsv                  (term list input)
│   │   ├── agent_list_2 - Blad1.tsv
│   │   ├── agent_list_3 - Blad1.tsv
│   │   ├── agent_list_4 - Blad1.tsv
│   │   ├── ngrams_frequencies.tsv             (generated)
│   │   ├── word_lemma_frequencies.tsv         (generated)
│   │   ├── agent - Blad1.expanded.tsv         (generated)
│   │   └── HitPyRank.ModeA.ListExample.tsv    (template)
│   │
│   └── Mode B/
│       ├── output first selection.xlsx        (4.4 MB)
│       ├── output second_selection.xlsx       (1.9 MB)
│       ├── output third selection.xlsx        (2.0 MB)
│       ├── output fourth selection.xlsx       (707 KB)
│       └── output fifth selection.xlsx        (265 KB)
│
├── Articles/                                  (89 MB - 20 PDF files)
├── Cleaned text files/                        (11 MB - 368 .txt files)
├── FoLia files/                              (227 MB - 386 .xml files)
│
├── Finalizing the corpus/
│   ├── Corpus after first and second selection.xlsx
│   ├── Final corpus.xlsx
│   ├── First selection corpus.xlsx
│   └── Second selection corpus.xlsx
│
├── Results annotation/
│   ├── Annotation Corpus.xlsx
│   ├── Annotation results agent system.xlsx
│   └── Annotation results no agent system.xlsx
│
├── ground truth/
│   ├── Annotation Ground Truth.xlsx
│   ├── Annotation Model.pdf
│   └── Corpus ground truth.pdf
│
└── [Additional documentation files]
    ├── instruction manual cleaning text files in visual studio code.rtf
    ├── Corpus paragraphs.docx
    └── Articles not included in the corpus.pdf
```

---

## Core Code Files

### 1. HitPyRank.v1.py (469 lines)

**Location:** `/HitPyRank/Mode A/HitPyRank.v1.py`

**Purpose:** Main NLP text processing tool for corpus analysis and paragraph extraction

**Architecture:** Interactive console application with 2 operating modes

#### Mode A - N-gram Frequency Extraction & Term Expansion
- Parses FoLiA XML format corpus files
- Extracts unigrams, bigrams, trigrams, and 4-grams from sentences
- Generates frequency lists (TSV format):
  - `ngrams_frequencies.tsv`: Raw n-gram frequencies
  - `word_lemma_frequencies.tsv`: Word-lemma pairs with frequencies
- Implements **wildcard expansion** mechanism:
  - Supports pattern matching (e.g., `autonom*` expands to autonomous, autonomy, autonomously, etc.)
  - Supports lemma-based expansion (e.g., expand lemma "agent" to all word forms)
  - Supports n-gram matching with word boundaries

**Key Functions:**
- `generate_ngrams(words, n)`: Creates n-grams from word lists
- `expansion_writer(inputrow, ngramrow, filepath)`: Appends expansion data to lists
- Ranking system (0-3): Rank 0 expands but isn't used for retrieval; ranks 1-3 used for paragraph extraction

#### Mode B - Paragraph Extraction
- Loads expanded term lists (from Mode A output)
- Searches FoLiA XML corpus for paragraphs containing specific terms
- Performs exact word matching with regex boundaries: `\b{term}\b`
- Creates comprehensive output TSV with:
  - Paragraph ID, original ID, label, hit terms, frequencies, ranking
  - Full summary: `term.rank.list.count` format
  - Unique hit summary: aggregated counts per rank per list
  - Original paragraph text

**Data Structures Used:**
- `Counter` objects for frequency tracking
- `xml.etree.ElementTree` for XML parsing
- CSV reading/writing with tab delimiters
- Nested dictionaries for paragraph metadata

**Dependencies:** `folia`, `collections`, `re`, `glob`, `xml.etree`, `csv`, `numpy`

---

### 2. Pipeline Processing Scripts (Steps 1-6)

All scripts are Python files without `.py` extension.

#### STEP 1: Downloading results search queries from ArXiv
- Uses arXiv REST API: `http://export.arxiv.org/api/query`
- Handles pagination (max 1501 results per request)
- Parses XML response to extract paper IDs
- Saves results as text files: `paper_ids_{TERM1}_{TERM2}_{TERM3}.txt`
- Each of 50 search query combinations runs separately

#### STEP 2: Sorting the results of the search queries
- Uses pandas to read Excel without headers
- Sorts by first column (URLs)
- Exports sorted Excel file

#### STEP 3: Removing and counting duplicates
- Reads Excel file (no headers)
- Creates duplicate count column
- Removes duplicates keeping first occurrence
- Saves output: `unique_{filename}.xlsx`

#### STEP 4: Sorting Output step 3
- Sorts Excel by second column (duplicate count) in ascending order

#### STEP 5: Finding most common results in search queries, step 1
- Creates workbook with one sheet per search query text file
- Each sheet contains paper IDs from that specific search

#### STEP 6: Finding most common results in search queries
- Compares papers across search queries
- Marks papers found in each search with 'x'
- Creates Excel with papers as rows, searches as columns

---

### 3. Data Selection Code

#### Extract colored rows
- Reads Excel file with openpyxl
- Identifies rows by cell coloring (visual filtering)
- Extracts uncolored rows (yellow/highlighted = excluded)
- Converts to filtered Excel file

---

## Data Formats

### Input: Search Query Combinations
**Location:** `/Files used in extracting the corpus/STEP 1: Input - Complete list of search combinations`

**Content:** 50 search specifications
```
agent --- action --- reinforcement learning
agent --- prompt --- reinforcement learning
...
agent --- policy --- AGI
```

**Attributes (10):** action, prompt, goal, history, memory, module, reward, state, environment, policy

**System Types (5):** reinforcement, autonomous, LLM, simulation, AGI

### Intermediate: FoLiA XML
**Location:** `/FoLia files/`

**Structure:** Hierarchical XML with metadata, tokens, sentences, paragraphs
```xml
<FoLiA>
  <metadata> - provenance, processors, language
  <text>
    <p> (paragraph)
      <s> (sentence)
        <w> (word/token)
          <t> (text)
          <lemma class="...">
```
- Created by `ucto` tokenizer processor
- Enables linguistic analysis (lemmatization, POS tagging ready)

### Output: HitPyRank Term Lists
**Format:** TSV with columns for term, rank, frequency, list name, expanded forms

---

## Data Flow Architecture

```
PHASE 1: DATA ACQUISITION (Steps 1-6)
├─ Input: 50 search query combinations
├─ Process: Query arXiv → aggregate → deduplicate → cross-reference
└─ Output: 50 text files with paper IDs

PHASE 2: TEXT PREPROCESSING
├─ Input: Selected arXiv PDFs
├─ Process: PDF → plain text → FoLiA XML (tokenization)
└─ Output: 386 FoLiA XML files (227 MB annotated corpus)

PHASE 3: LINGUISTIC PROCESSING (HitPyRank Mode A)
├─ Input: FoLiA XML corpus + manual term lists
├─ Process: Extract n-grams & lemmas → expand wildcards
└─ Output: Frequency lists + expanded term lists

PHASE 4: PARAGRAPH EXTRACTION (HitPyRank Mode B)
├─ Input: Expanded term lists + FoLiA XML corpus
├─ Process: Exact word matching → frequency counting → ranking
└─ Output: TSV with ranked paragraphs (5 selection iterations)

PHASE 5: MANUAL CURATION & EVALUATION
├─ Input: Auto-extracted paragraphs
├─ Process: Manual color-coding in Excel → automatic extraction
└─ Output: Final refined corpus with annotations & ground truth
```

---

## Corpus Statistics

- **Total papers searched:** ~50 × variable results (max 1,500 per query)
- **Unique papers after deduplication:** ~386 papers (FoLiA files)
- **Total clean text extracted:** 11 MB (368 files)
- **Annotated corpus size:** 227 MB (FoLiA XML)
- **Final extracted paragraphs:** From ~4.4 MB (10,000+ initially) down to 265 KB (final selection)
- **Search combinations:** 50 (10 agent attributes × 5 system types)
- **Selection iterations:** 5 progressive filtering rounds

---

## Known Issues & Limitations

### Code Quality Issues
1. **No file extensions:** Python scripts lack `.py` extension
2. **No input validation:** No verification that corpus exists or contains FoLiA files
3. **Minimal error handling:** Assumes well-formed input
4. **Hardcoded paths:** Configuration embedded in scripts
5. **No modularization:** Each step is a separate standalone script
6. **No CLI interface:** Mode selection via interactive input

### Architectural Issues
1. **Tightly coupled:** Steps depend on specific file formats/locations
2. **No configuration file:** Settings scattered across scripts
3. **Manual steps required:** PDF→text and text→FoLiA not automated
4. **No logging:** Progress tracking not implemented
5. **No tests:** No unit or integration tests
6. **No dependency management:** No requirements.txt

### Functional Limitations
1. **Wildcard limitations:** Cannot handle wildcards in middle of words (e.g., `a*nt`)
2. **POS tagging:** Ready for POS tags but not fully implemented
3. **Character restrictions:** Special characters and numbers not properly handled
4. **Ranking enforcement:** Ranking range (0-3) not validated on input
5. **Empty corpus:** Doesn't warn if corpus folder is empty

---

## Dependencies

**Python Libraries:**
- `folia` - FoLiA XML parsing
- `pandas` - Excel/data manipulation
- `openpyxl` - Advanced Excel operations
- `requests` - HTTP requests (arXiv API)
- `csv` - CSV reading/writing
- `xml.etree.ElementTree` - XML parsing
- `collections.Counter` - Frequency counting
- `numpy` - Array operations
- `re` - Regular expressions
- `glob` - File pattern matching
- `os` - File system operations

**External Tools:**
- `ucto` - Tokenizer for FoLiA generation
- PDF text extraction (manual/external tool)

---

## Notes

- All code was created with GPT-3.5 assistance by a non-programmer researcher
- The project follows the "Concepts in Motion" methodology
- Focus is on agent-related concepts in AI/CS literature
