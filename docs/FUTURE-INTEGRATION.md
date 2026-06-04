# Future Integration: ternary-language

## Current State
Provides `TernaryTokenizer` (whitespace/punctuation splitting with ternary sentiment), `SentimentClassifier` (aggregate token sentiment with confidence scoring), `TernaryGrammar` for production rules on ternary tokens, a `TernaryParser`, and a language model with ternary state transitions.

## Integration Opportunities

### With ternary-cell / construct-core
Room descriptions become ternary language. When `construct-core` needs to describe a room's state for inter-agent communication, it serializes the cell grid into a ternary token sequence. `SentimentClassifier::classify()` then produces a room-level sentiment: positive (comfortable/operational), neutral (nominal), negative (problematic). This becomes the cell's broadcast signal — a single trit summarizing room health for neighboring cells.

### With ternary-protocol
The `TernaryGrammar`'s production rules define the protocol grammar. Each message type in `ternary-protocol` is a production rule. The `TernaryParser` validates incoming messages against the grammar — malformed packets are rejected at parse time rather than runtime. `classify_with_confidence()` provides message importance scoring for priority routing.

### With ternary-language-model
The language model's ternary state transitions can predict the next room state given a sequence of room descriptions. This is a lightweight room-state predictor that runs on ESP32: observe three ternary tokens of room state, predict the fourth.

## Potential in Mature Systems
In PLATO, the `TernaryTokenizer` becomes the universal serialization layer. All inter-construct communication passes through ternary tokenization. The `add_word()` method allows domain-specific vocabulary injection — a room managing a kitchen adds culinary terms with associated ternary sentiment. The `TernaryGrammar` enforces communication protocols at the type level: only grammatically valid messages propagate.

## Cross-Pollination Ideas
**Music × Language:** Ternary grammar productions map to musical phrase structure. A production rule `S → A B C` becomes a phrase decomposed into three motifs. The ternary sentiment of each token maps to tension/release in music theory. This connects to `ternary-music` and `counterpoint-engine-rs`.

**Linguistic polyformalism × Language:** The 7 constraint types from `linguistic-polyformalism-shell` could define grammar constraint annotations. Each production rule carries a constraint type, enabling multi-formal validation.

## Dependencies for Next Steps
- `ternary-protocol` message format must align with `TernaryGrammar` production rules
- Benchmark tokenizer throughput for real-time room-state encoding
- Vocabulary learning: can `add_word()` be automated from room state observations?
