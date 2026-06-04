#![forbid(unsafe_code)]

//! Language and grammar processing with ternary sentiment.
//!
//! TernaryTokenizer, SentimentClassifier, TernaryGrammar,
//! TernaryParser, and language model with ternary state transitions.

use std::collections::HashMap;

/// Ternary sentiment label.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Sentiment {
    Negative = -1,
    Neutral = 0,
    Positive = 1,
}

impl Sentiment {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Sentiment::Negative),
            0 => Some(Sentiment::Neutral),
            1 => Some(Sentiment::Positive),
            _ => None,
        }
    }
}

/// Token with an associated ternary value.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TernaryToken {
    pub text: String,
    pub value: Sentiment,
}

impl TernaryToken {
    pub fn new(text: &str, value: Sentiment) -> Self {
        Self { text: text.to_string(), value }
    }
}

/// Simple tokenizer that splits on whitespace and punctuation, assigning ternary values.
pub struct TernaryTokenizer {
    positive_words: HashMap<String, Sentiment>,
    negative_words: HashMap<String, Sentiment>,
}

impl TernaryTokenizer {
    pub fn new() -> Self {
        let mut pos = HashMap::new();
        let mut neg = HashMap::new();
        for w in &["good", "great", "excellent", "happy", "love", "wonderful", "best", "amazing", "awesome", "fantastic"] {
            pos.insert(w.to_string(), Sentiment::Positive);
        }
        for w in &["bad", "terrible", "awful", "sad", "hate", "worst", "horrible", "poor", "angry", "disappointing"] {
            neg.insert(w.to_string(), Sentiment::Negative);
        }
        Self { positive_words: pos, negative_words: neg }
    }

    pub fn tokenize(&self, text: &str) -> Vec<TernaryToken> {
        text.split_whitespace()
            .map(|w| {
                let cleaned: String = w.chars().filter(|c| c.is_alphabetic()).collect::<String>().to_lowercase();
                let sent = if self.positive_words.contains_key(&cleaned) {
                    Sentiment::Positive
                } else if self.negative_words.contains_key(&cleaned) {
                    Sentiment::Negative
                } else {
                    Sentiment::Neutral
                };
                TernaryToken::new(&cleaned, sent)
            })
            .collect()
    }

    pub fn add_word(&mut self, word: &str, sentiment: Sentiment) {
        match sentiment {
            Sentiment::Positive => { self.positive_words.insert(word.to_lowercase(), sentiment); }
            Sentiment::Negative => { self.negative_words.insert(word.to_lowercase(), sentiment); }
            Sentiment::Neutral => {}
        }
    }
}

/// Sentiment classifier combining token-level signals.
pub struct SentimentClassifier {
    tokenizer: TernaryTokenizer,
}

impl SentimentClassifier {
    pub fn new() -> Self {
        Self { tokenizer: TernaryTokenizer::new() }
    }

    pub fn with_tokenizer(tokenizer: TernaryTokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn classify(&self, text: &str) -> Sentiment {
        let tokens = self.tokenizer.tokenize(text);
        let score: i32 = tokens.iter().map(|t| t.value as i32).sum();
        if score > 0 {
            Sentiment::Positive
        } else if score < 0 {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        }
    }

    /// Classify with a confidence score (magnitude / total tokens).
    pub fn classify_with_confidence(&self, text: &str) -> (Sentiment, f64) {
        let tokens = self.tokenizer.tokenize(text);
        if tokens.is_empty() {
            return (Sentiment::Neutral, 0.0);
        }
        let score: i32 = tokens.iter().map(|t| t.value as i32).sum();
        let confidence = (score.abs() as f64) / (tokens.len() as f64);
        let sent = if score > 0 { Sentiment::Positive } else if score < 0 { Sentiment::Negative } else { Sentiment::Neutral };
        (sent, confidence)
    }
}

/// A ternary grammar symbol.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TernarySymbol {
    Terminal(String),
    NonTerminal(String),
    Ternary(i8), // -1, 0, 1
}

/// A production rule: LHS -> RHS.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ProductionRule {
    pub lhs: TernarySymbol,
    pub rhs: Vec<TernarySymbol>,
}

impl ProductionRule {
    pub fn new(lhs: TernarySymbol, rhs: Vec<TernarySymbol>) -> Self {
        Self { lhs, rhs }
    }
}

/// Ternary grammar with production rules.
pub struct TernaryGrammar {
    rules: Vec<ProductionRule>,
    start: TernarySymbol,
}

impl TernaryGrammar {
    pub fn new(start: TernarySymbol) -> Self {
        Self { rules: Vec::new(), start }
    }

    pub fn add_rule(&mut self, rule: ProductionRule) {
        self.rules.push(rule);
    }

    pub fn rules_for(&self, sym: &TernarySymbol) -> Vec<&ProductionRule> {
        self.rules.iter().filter(|r| &r.lhs == sym).collect()
    }

    pub fn rules(&self) -> &[ProductionRule] {
        &self.rules
    }

    pub fn start(&self) -> &TernarySymbol {
        &self.start
    }

    /// Check if a symbol is terminal (has no rules).
    pub fn is_terminal(&self, sym: &TernarySymbol) -> bool {
        self.rules_for(sym).is_empty()
    }

    /// Generate all strings of depth <= max_depth from start symbol.
    pub fn generate(&self, max_depth: usize) -> Vec<String> {
        self.generate_from(&self.start.clone(), max_depth)
    }

    fn generate_from(&self, sym: &TernarySymbol, depth: usize) -> Vec<String> {
        if depth == 0 || self.is_terminal(sym) {
            match sym {
                TernarySymbol::Terminal(s) => vec![s.clone()],
                TernarySymbol::Ternary(v) => vec![format!("T({})", v)],
                TernarySymbol::NonTerminal(_) => vec![],
            }
        } else {
            let mut results = Vec::new();
            for rule in self.rules_for(sym) {
                let mut partials = vec![String::new()];
                for rhs_sym in &rule.rhs {
                    let expansions = self.generate_from(rhs_sym, depth - 1);
                    let mut new_partials = Vec::new();
                    for p in &partials {
                        for e in &expansions {
                            let s = if p.is_empty() { e.clone() } else { format!("{} {}", p, e) };
                            new_partials.push(s);
                        }
                    }
                    partials = new_partials;
                }
                results.extend(partials);
            }
            results
        }
    }
}

/// A parse tree node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseNode {
    pub symbol: TernarySymbol,
    pub children: Vec<ParseNode>,
}

impl ParseNode {
    pub fn terminal(text: &str) -> Self {
        Self { symbol: TernarySymbol::Terminal(text.to_string()), children: Vec::new() }
    }

    pub fn non_terminal(name: &str, children: Vec<ParseNode>) -> Self {
        Self { symbol: TernarySymbol::NonTerminal(name.to_string()), children }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collect all terminal strings.
    pub fn terminals(&self) -> Vec<String> {
        if self.is_leaf() {
            match &self.symbol {
                TernarySymbol::Terminal(s) => vec![s.clone()],
                TernarySymbol::Ternary(v) => vec![format!("T({})", v)],
                _ => vec![],
            }
        } else {
            self.children.iter().flat_map(|c| c.terminals()).collect()
        }
    }
}

/// Recursive descent parser for TernaryGrammar.
pub struct TernaryParser {
    grammar: TernaryGrammar,
}

impl TernaryParser {
    pub fn new(grammar: TernaryGrammar) -> Self {
        Self { grammar }
    }

    /// Try to parse input tokens against the grammar's start symbol.
    pub fn parse(&self, tokens: &[String]) -> Option<ParseNode> {
        self.parse_symbol(&self.grammar.start().clone(), tokens, 0).and_then(|(node, pos)| {
            if pos == tokens.len() { Some(node) } else { None }
        })
    }

    fn parse_symbol(&self, sym: &TernarySymbol, tokens: &[String], pos: usize) -> Option<(ParseNode, usize)> {
        match sym {
            TernarySymbol::Terminal(s) => {
                if pos < tokens.len() && &tokens[pos] == s {
                    Some((ParseNode::terminal(s), pos + 1))
                } else {
                    None
                }
            }
            TernarySymbol::Ternary(v) => {
                let s = format!("T({})", v);
                if pos < tokens.len() && tokens[pos] == s {
                    Some((ParseNode::terminal(&s), pos + 1))
                } else {
                    None
                }
            }
            TernarySymbol::NonTerminal(_) => {
                for rule in self.grammar.rules_for(sym) {
                    if let Some((node, new_pos)) = self.try_rule(rule, tokens, pos) {
                        return Some((node, new_pos));
                    }
                }
                None
            }
        }
    }

    fn try_rule(&self, rule: &ProductionRule, tokens: &[String], pos: usize) -> Option<(ParseNode, usize)> {
        let mut children = Vec::new();
        let mut current_pos = pos;
        for sym in &rule.rhs {
            match self.parse_symbol(sym, tokens, current_pos) {
                Some((child, new_pos)) => {
                    children.push(child);
                    current_pos = new_pos;
                }
                None => return None,
            }
        }
        Some((ParseNode { symbol: rule.lhs.clone(), children }, current_pos))
    }
}

/// Ternary language model with state transitions.
pub struct TernaryLanguageModel {
    /// (from_state, to_state) -> count
    transitions: HashMap<(i8, i8), usize>,
    /// state -> count
    state_counts: HashMap<i8, usize>,
    /// initial state counts
    initial_counts: HashMap<i8, usize>,
    total_initial: usize,
}

impl TernaryLanguageModel {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            state_counts: HashMap::new(),
            initial_counts: HashMap::new(),
            total_initial: 0,
        }
    }

    /// Train on a sequence of ternary states.
    pub fn train(&mut self, sequence: &[i8]) {
        if sequence.is_empty() { return; }
        // Validate
        for &s in sequence {
            if s != -1 && s != 0 && s != 1 { return; }
        }
        *self.initial_counts.entry(sequence[0]).or_insert(0) += 1;
        self.total_initial += 1;
        for &s in sequence {
            *self.state_counts.entry(s).or_insert(0) += 1;
        }
        for window in sequence.windows(2) {
            *self.transitions.entry((window[0], window[1])).or_insert(0) += 1;
        }
    }

    /// Probability of transitioning from one state to another.
    pub fn transition_prob(&self, from: i8, to: i8) -> f64 {
        let from_count = self.state_counts.get(&from).copied().unwrap_or(0);
        if from_count == 0 { return 1.0 / 3.0; }
        let trans_count = self.transitions.get(&(from, to)).copied().unwrap_or(0);
        trans_count as f64 / from_count as f64
    }

    /// Initial probability of a state.
    pub fn initial_prob(&self, state: i8) -> f64 {
        if self.total_initial == 0 { return 1.0 / 3.0; }
        self.initial_counts.get(&state).copied().unwrap_or(0) as f64 / self.total_initial as f64
    }

    /// Log probability of an entire sequence.
    pub fn sequence_log_prob(&self, sequence: &[i8]) -> f64 {
        if sequence.is_empty() { return 0.0; }
        let mut lp = self.initial_prob(sequence[0]).ln();
        for window in sequence.windows(2) {
            let p = self.transition_prob(window[0], window[1]);
            lp += p.ln();
        }
        lp
    }

    /// Most likely next state given current state.
    pub fn most_likely_next(&self, current: i8) -> i8 {
        let mut best = 0i8;
        let mut best_p = -1.0f64;
        for &s in &[-1i8, 0, 1] {
            let p = self.transition_prob(current, s);
            if p > best_p {
                best_p = p;
                best = s;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_from_i8() {
        assert_eq!(Sentiment::from_i8(-1), Some(Sentiment::Negative));
        assert_eq!(Sentiment::from_i8(0), Some(Sentiment::Neutral));
        assert_eq!(Sentiment::from_i8(1), Some(Sentiment::Positive));
        assert_eq!(Sentiment::from_i8(5), None);
    }

    #[test]
    fn test_tokenizer_positive() {
        let tok = TernaryTokenizer::new();
        let tokens = tok.tokenize("I love this great day");
        let pos_count = tokens.iter().filter(|t| t.value == Sentiment::Positive).count();
        assert!(pos_count >= 2);
    }

    #[test]
    fn test_tokenizer_negative() {
        let tok = TernaryTokenizer::new();
        let tokens = tok.tokenize("This is terrible and awful");
        let neg_count = tokens.iter().filter(|t| t.value == Sentiment::Negative).count();
        assert!(neg_count >= 2);
    }

    #[test]
    fn test_tokenizer_neutral() {
        let tok = TernaryTokenizer::new();
        let tokens = tok.tokenize("the cat sat on the mat");
        assert!(tokens.iter().all(|t| t.value == Sentiment::Neutral));
    }

    #[test]
    fn test_tokenizer_custom_word() {
        let mut tok = TernaryTokenizer::new();
        tok.add_word("rad", Sentiment::Positive);
        let tokens = tok.tokenize("that was rad");
        assert!(tokens.iter().any(|t| t.text == "rad" && t.value == Sentiment::Positive));
    }

    #[test]
    fn test_classifier_positive_text() {
        let cls = SentimentClassifier::new();
        assert_eq!(cls.classify("I love this amazing wonderful day"), Sentiment::Positive);
    }

    #[test]
    fn test_classifier_negative_text() {
        let cls = SentimentClassifier::new();
        assert_eq!(cls.classify("I hate this terrible awful day"), Sentiment::Negative);
    }

    #[test]
    fn test_classifier_neutral_text() {
        let cls = SentimentClassifier::new();
        assert_eq!(cls.classify("the cat sat"), Sentiment::Neutral);
    }

    #[test]
    fn test_classifier_confidence() {
        let cls = SentimentClassifier::new();
        let (sent, conf) = cls.classify_with_confidence("great great great");
        assert_eq!(sent, Sentiment::Positive);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_classifier_empty() {
        let cls = SentimentClassifier::new();
        let (sent, conf) = cls.classify_with_confidence("");
        assert_eq!(sent, Sentiment::Neutral);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_grammar_basic() {
        let mut g = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
        g.add_rule(ProductionRule::new(
            TernarySymbol::NonTerminal("S".into()),
            vec![TernarySymbol::Terminal("hello".into())],
        ));
        assert_eq!(g.rules().len(), 1);
        let gen = g.generate(2);
        assert!(gen.contains(&"hello".to_string()));
    }

    #[test]
    fn test_grammar_is_terminal() {
        let mut g = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
        g.add_rule(ProductionRule::new(
            TernarySymbol::NonTerminal("S".into()),
            vec![TernarySymbol::Terminal("a".into())],
        ));
        assert!(g.is_terminal(&TernarySymbol::Terminal("a".into())));
        assert!(!g.is_terminal(&TernarySymbol::NonTerminal("S".into())));
    }

    #[test]
    fn test_grammar_ternary_symbols() {
        let mut g = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
        g.add_rule(ProductionRule::new(
            TernarySymbol::NonTerminal("S".into()),
            vec![TernarySymbol::Ternary(1)],
        ));
        let gen = g.generate(2);
        assert!(gen.contains(&"T(1)".to_string()));
    }

    #[test]
    fn test_parser_simple() {
        let mut g = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
        g.add_rule(ProductionRule::new(
            TernarySymbol::NonTerminal("S".into()),
            vec![TernarySymbol::Terminal("a".into()), TernarySymbol::Terminal("b".into())],
        ));
        let parser = TernaryParser::new(g);
        let result = parser.parse(&["a".to_string(), "b".to_string()]);
        assert!(result.is_some());
        let tree = result.unwrap();
        assert_eq!(tree.terminals(), vec!["a", "b"]);
    }

    #[test]
    fn test_parser_fail() {
        let mut g = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
        g.add_rule(ProductionRule::new(
            TernarySymbol::NonTerminal("S".into()),
            vec![TernarySymbol::Terminal("a".into())],
        ));
        let parser = TernaryParser::new(g);
        assert!(parser.parse(&["b".to_string()]).is_none());
    }

    #[test]
    fn test_parser_partial_fail() {
        let mut g = TernaryGrammar::new(TernarySymbol::NonTerminal("S".into()));
        g.add_rule(ProductionRule::new(
            TernarySymbol::NonTerminal("S".into()),
            vec![TernarySymbol::Terminal("a".into())],
        ));
        let parser = TernaryParser::new(g);
        assert!(parser.parse(&["a".to_string(), "b".to_string()]).is_none());
    }

    #[test]
    fn test_language_model_train_and_query() {
        let mut lm = TernaryLanguageModel::new();
        lm.train(&[1, 1, 1, 1]);
        assert!(lm.transition_prob(1, 1) > 0.0);
        assert_eq!(lm.most_likely_next(1), 1);
    }

    #[test]
    fn test_language_model_initial_prob() {
        let mut lm = TernaryLanguageModel::new();
        lm.train(&[1, 0, -1]);
        lm.train(&[1, 0, 0]);
        assert!(lm.initial_prob(1) > 0.4);
    }

    #[test]
    fn test_language_model_sequence_prob() {
        let mut lm = TernaryLanguageModel::new();
        lm.train(&[1, 1, 0]);
        lm.train(&[0, 0, -1]);
        let p1 = lm.sequence_log_prob(&[1, 1, 0]);
        let p2 = lm.sequence_log_prob(&[-1, -1, -1]);
        assert!(p1 > p2);
    }

    #[test]
    fn test_language_model_empty() {
        let lm = TernaryLanguageModel::new();
        assert_eq!(lm.sequence_log_prob(&[]), 0.0);
        assert!((lm.initial_prob(0) - 1.0/3.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_node_terminals() {
        let node = ParseNode::non_terminal("S", vec![
            ParseNode::terminal("a"),
            ParseNode::terminal("b"),
        ]);
        assert_eq!(node.terminals(), vec!["a", "b"]);
        assert!(!node.is_leaf());
        assert!(node.children[0].is_leaf());
    }
}
