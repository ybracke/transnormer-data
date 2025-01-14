# Background

## Token replacements

Generally, when we replace tokens that have annotations, we have to replace their annotations as well.
However, often we do not have to provide the new annotations explicitly.

For example, I want to replace a token "daß" by "dass". My original token has the following properties:
```json
{"text" : "daß", "ws": false, "span": [0,3]}
```
Then I have to replace the annotations for "ws" and "spans" as well. However, it makes sense that I do not have to pass the values for the new "ws" and "span" explicitly and instead have a function that infers the new values - either from the old value (the new token should - generally - be preceded by the same whitespace as the old), or from a combination of the new token and the old value (the new span should have the start of the old and start+len(new) as the end; Note that this annotation affects the span annotation of the following tokens and, thus, these have to updated accordingly.).


### 1:1-Replacements

Example: replace "daß" with "dass"

Applying 1:1 token replacements leaves the tokenization and alignment intact. Therefore, we can simply apply the replacement to `target_tok` and propagate it to `target_raw` (by joining the tokens on their associated whitespace). No need for retokenization and recomputation of the alignments. We do have to recompute the token spans though!

### 1:n/n:1-Replacements

Example: replace ["zuviel"] with ["zu", "viel"]

Two approaches:
1. As string replacement in `target_raw`
2. As list modification in `target_tok`

Applying 1:n/n:1 token replacements changes the tokenization directly. We also have to reset spans, whitespace annotations and alignments. Given that we know the number of tokens in n as well as the number of characters, we should be able to reset these values with a less costly computation (basically just add +n to all folowing alignments, and +number_of_chars to all following spans). `target_raw` can be produced by joining the modified `target_tok` then.
