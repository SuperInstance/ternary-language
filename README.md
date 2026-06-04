# ternary-language

Language and grammar processing with ternary sentiment — tokenization, sentiment classification, context-free grammars, recursive-descent parsing, and Markov-chain language models over {-1, 0, +1} states.

## Why This Exists

NLP typically uses continuous embeddings and large models. But there's a useful niche where text processing maps directly onto ternary signals: sentiment (negative / neutral / positive), decision labels (reject / abstain / approve), and grammar symbols that carry ternary payloads. This crate provides self-contained, zero-dependency tools for ternary-aware text processing — from tokenization and sentiment scoring through full grammar-based parsing and statistical language modeling. `forbid(unsafe_code)` throughout.

## Core Concepts

- **Sentiment**: Ternary label — `Negative` (-1), `Neutral` (0), `Positive` (+1).
- **TernaryTokenizer**: Splits text into tokens, assigns sentiment based on a built-in or custom lexicon.
- **SentimentClassifier**: Aggregates token-level sentiment into document-level classification with confidence scores.
- **TernaryGrammar**: Context-free grammar with ternary-valued symbols; generate all strings up to a depth.
- **TernaryParser**: Recursive-descent parser for `TernaryGrammar`; produces `ParseNode` trees.
- **TernaryLanguageModel**: Markov-chain model over ternary states with transition probabilities, sequence scoring, and next-state prediction.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-language = "0.1"
```

```rust
use ternary_language::{
    SentimentClassifier, TernaryTokenizer, Sentiment,
    TernaryGrammar, TernarySymbol, ProductionRule, TernaryParser,
    TernaryLanguageModel,
};

fn main() {
    // Sentiment analysis
    let classifier = SentimentClassifier::new();
    assert_eq!(classifier.classify("I love this amazing day"), Sentiment::Positive);
    assert_eq!(classifier.classify("I hate this terrible weather"), Sentiment::Negative);

    let (sent, confidence) = classifier.classify_with_confidence("great great great");
    println!("Sentiment: {:?}, confidence: {:.2}", sent, confidence);

    // Custom vocabulary
    let mut tok = TernaryTokenizer::new();
    tok.add_word("rad", Sentiment::Positive);
    tok.add_word("meh", Sentiment::Neutral);

    // Grammar and parsing
    let mut grammar = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
    grammar.add_rule(ProductionRule::new(
        TernarySymbol::NonTerminal("S".into()),
        vec![
            TernarySymbol::Terminal("hello".into()),
            TernarySymbol::Terminal("world".into()),
        ],
    ));

    let parser = TernaryParser::new(grammar);
    let tree = parser.parse(&["hello".to_string(), "world".to_string()]).unwrap();
    assert_eq!(tree.terminals(), vec!["hello", "world"]);

    // Language model
    let mut lm = TernaryLanguageModel::new();
    lm.train(&[1, 1, 0, -1]);
    lm.train(&[1, 1, 1, 0]);
    println!("P(1→1) = {:.3}", lm.transition_prob(1, 1));
    println!("Most likely next after 1: {}", lm.most_likely_next(1));
}
```

## API Overview

| Type | Description |
|---|---|
| `Sentiment` | Ternary label: `Negative`, `Neutral`, `Positive` |
| `TernaryToken` | Token text + associated sentiment |
| `TernaryTokenizer` | Whitespace/punctuation tokenizer with sentiment lexicon |
| `SentimentClassifier` | Aggregate token sentiment with `classify()` and `classify_with_confidence()` |
| `TernarySymbol` | Grammar symbol: `Terminal`, `NonTerminal`, or `Ternary(i8)` |
| `ProductionRule` | LHS → RHS rewrite rule |
| `TernaryGrammar` | CFG with `add_rule()`, `generate(max_depth)`, `is_terminal()` |
| `ParseNode` | Tree node with `terminals()` for leaf extraction |
| `TernaryParser` | Recursive-descent parser producing `ParseNode` |
| `TernaryLanguageModel` | Markov chain over {-1, 0, 1} with `train()`, `transition_prob()`, `sequence_log_prob()`, `most_likely_next()` |

## How It Works

**Tokenization** splits on whitespace, strips punctuation, lowercases, and looks up each word in positive/negative lexicons (built-in defaults, extensible via `add_word`). Unrecognized words get `Neutral`.

**Classification** sums token sentiment values (cast to `i32`) and maps the total: positive sum → `Positive`, negative → `Negative`, zero → `Neutral`. Confidence is `|sum| / token_count`.

**Grammar** uses a standard CFG representation. `generate(max_depth)` recursively expands the start symbol, collecting all terminal strings up to the given depth. `TernarySymbol::Ternary(v)` embeds raw ternary values in the grammar.

**Parsing** is recursive descent: try each production rule for a non-terminal in order, backtrack on failure. Returns a `ParseNode` tree on success, `None` on failure.

**Language modeling** counts state transitions in training sequences and estimates `P(to | from)` via maximum likelihood. Sequence log-probability chains initial and transition probabilities. `most_likely_next` simply picks the highest-probability successor.

## Use Cases

- **Lightweight sentiment analysis**: Classify customer feedback, reviews, or social media posts without a large model.
- **Grammar-based ternary signal parsing**: Define grammars over ternary symbols and parse structured ternary data streams.
- **Ternary sequence prediction**: Train a Markov model on ternary time series (e.g., stock direction signals) and predict next state.
- **Custom lexicon scoring**: Build domain-specific sentiment lexicons for specialized text analysis.

## Ecosystem

Part of the **SuperInstance** ternary computing suite:

- `ternary-lattice` — lattice structures for ternary values
- `ternary-codes` — error-correcting codes for ternary data
- `ternary-gradient` — gradient-free optimization on ternary landscapes
- `ternary-language` — this crate
- `ternary-trees` — ternary decision trees and forests
- `ternary-transform` — wavelet, Fourier, and kernel transforms
- `ternary-planning` — planning and scheduling with ternary priorities
- `ternary-rl` — reinforcement learning with ternary actions
- `ternary-som` — self-organizing maps for ternary data
- `ternary-failure` — failure analysis with ternary classification

## License

MIT
