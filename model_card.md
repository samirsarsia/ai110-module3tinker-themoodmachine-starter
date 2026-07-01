# Model Card: Mood Machine

This model card documents the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

I built and compared **both** versions on the same dataset.

## 1. Model Overview

**Model type:**
Both. A hand-written rule-based classifier (`mood_analyzer.py`) and a bag-of-words logistic-regression classifier (`ml_experiments.py`), trained and evaluated on the same labeled data.

**Intended purpose:**
Classify short, social-media-style text posts into one of four moods: `positive`, `negative`, `neutral`, or `mixed`. It is an educational tool for exploring how rule-based and learned models differ, not a production sentiment system.

**How it works (brief):**
- **Rule based:** the text is tokenized, each token is checked against a positive/negative word list, positive words add +1 and negative words subtract 1, a negation word flips the sign of the next mood word, and the final numeric score is mapped to a label (`>0` positive, `<0` negative, `0` neutral).
- **ML:** posts are converted to bag-of-words count vectors (`CountVectorizer`), and a `LogisticRegression` model learns which words are associated with which label directly from the labeled examples.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` (in `dataset.py`) started with 6 starter posts. I expanded it in two rounds to **18 posts** total:
- Round 1 (+8): realistic social-media language — slang (`lowkey`, `no cap`), emojis (😂🔥 😭💅 💀 🙄 🥲), sarcasm (`oh great, another Monday. can't wait 🙄`), and mixed feelings (`cried a little but honestly needed that`).
- Round 2 (+4): additional sarcasm, a subtle-positive (`honestly today was pretty solid, no complaints`), an explicit mixed emotion (`nervous about the interview but also kinda hyped ngl`), and a plainly negative review (`the food was cold and the service was slow`).

`TRUE_LABELS` holds one label per post; both lists are length 18 and verified aligned. Label spread: ~5 positive, ~6 negative, ~3 neutral, ~4 mixed.

**Labeling process:**
I labeled by the *intended* meaning a human reader would infer, not the surface words. For sarcasm I labeled the true intent (e.g. `oh great... 🙄` → `negative`, despite the word "great"). For posts holding two opposing feelings at once I used `mixed`.

Posts that were genuinely hard to label (documented as edge cases):
- `"meeting got moved again lol whatever"` — labeled `neutral`, but a case exists for `negative` (mild annoyance).
- `"idk how i feel about this tbh 🥲"` — labeled `neutral`, but could be `mixed` depending on how 🥲 is read.
- `"lowkey exhausted but we move 😮‍💨"` — labeled `mixed` (tired + resilient), but a reader could call it `negative`.

**Important characteristics of the dataset:**
- Contains slang and emojis.
- Includes multiple sarcastic posts.
- Several posts express genuinely mixed feelings.
- Posts are short and often ambiguous.

**Possible issues with the dataset:**
- **Small** (18 posts, ~3–6 per class) — far too little to train or fairly evaluate an ML model.
- **Class imbalance** across the four labels.
- **Skewed toward hard cases** (sarcasm, slang, emojis) that I deliberately wrote to stress the models — so accuracy here is a pessimistic estimate for "easy" text.
- **Culturally narrow slang** — mostly current English-language internet slang (see §7).

## 3. How the Rule Based Model Works

**Preprocessing (`preprocess`):**
- Lowercases and strips whitespace.
- Splits each emoji into its own token (so `"day 😂🔥"` → `day`, `😂`, `🔥`).
- Strips edge punctuation off words (`"exam!!"` → `exam`, `"fine."` → `fine`) while keeping internal apostrophes (`can't`, `it's` stay whole).

**Scoring rules (`score_text`):**
- Each token in `POSITIVE_WORDS` adds +1; each token in `NEGATIVE_WORDS` subtracts 1.
- **Enhancement — negation handling:** a negator (`not`, `never`, `no`, `don't`, `can't`) sets a flag that flips the polarity of the *next* mood word. So `"not happy"` scores −1 instead of +1, and `"not bad"` scores +1 instead of −1.
- **Vocabulary expansion:** I added `best`, `proud`, `hopeful`, and positive slang `fire`, `wicked` to `POSITIVE_WORDS` after breaker testing showed slang words scored 0.

**Label thresholds (`predict_label`):**
`score > 0` → positive, `score < 0` → negative, `score == 0` → neutral. I deliberately kept the simple thresholds: all non-zero scores in this dataset are exactly ±1, so requiring a stronger score (e.g. `>= 2`) would collapse everything to neutral and *lower* accuracy.

**Strengths:**
- Fully transparent — `explain()` shows exactly which words drove each decision.
- Reliable on clear, literal text (e.g. `"I love this, it is amazing and fun"` → +3 positive).
- Negation works correctly on standard grammar (`"this is not good"` → negative).

**Weaknesses:** sarcasm, unfamiliar slang, emoji sentiment, and the `mixed` label (see §6 for specific misclassified examples).

## 4. How the ML Model Works

**Features used:** Bag of words via `CountVectorizer` (word-count vectors; no math tuning).

**Training data:** Trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:** When I grew the dataset from 14 to 18 posts, the ML model's reported accuracy stayed pinned at **1.00**, while the rule-based accuracy *dropped* from 0.50 to 0.39. This is a red flag, not a success — see §5.

**Strengths:**
- Learns word→label associations automatically; no hand-maintained word lists.
- Can output `mixed`, which the rule model structurally cannot.

**Weaknesses:**
- **Evaluated on its own training data** (`ml_experiments.py` trains and tests on the same 18 posts — the code comment even says `# training accuracy`). With more features than examples, logistic regression can essentially memorize the labels, so 1.00 measures memorization, not generalization.
- Would likely pick up **spurious cues** (e.g. associating a specific emoji with a label because one training post used it) and has never been tested on unseen text.

## 5. Evaluation

**How I evaluated:**
Both models predict on all posts in `dataset.py` and report accuracy. Critically, these are **not** the same evaluation regime: the rule-based rules were written *without* fitting to the labels (honest), while the ML model is scored on the exact posts it trained on (optimistic).

**Final numbers (18-post dataset):**
| Model | Accuracy | What it measures |
|---|---|---|
| Rule based | **0.39** (7/18) | Honest — logic is fixed, never fit to labels |
| ML (as written) | **1.00** (18/18) | Training accuracy — saw the answers |

**Examples of correct predictions:**
- Rule based: `"Today was a terrible day"` → negative (word `terrible` = −1). Correct and explainable.
- Rule based: `"I am not happy about this"` → negative — negation flipped `happy` from +1 to −1. This is the negation enhancement working.
- ML: `"nervous about the interview but also kinda hyped ngl"` → mixed. The rule model can never produce `mixed`; the ML model learned it (though only because it saw this exact labeled post).

**Examples of incorrect predictions (and why they differ):**
- `"no cap this was the best day ever 😂🔥"` — **rule → negative** (wrong), **ML → positive** (right on training data). The rule model treats slang `no` as a negator, flipping `best` from +1 to −1. The ML model just memorized the post's words → positive.
- `"oh great, another Monday. can't wait 🙄"` — **rule → positive** (wrong), **ML → negative** (right on training data). The rule model reads `great` literally (+1); it has no way to detect the sarcasm the 🙄 conveys.

The failure modes differ: the rule model fails **systematically** on categories (all sarcasm, all `mixed`), while the ML model shows **no failures on training data** but has untested generalization.

## 6. Limitations

Specific, reproduced examples from evaluation and breaker testing:

- **Sarcasm read literally.** `"oh great, another Monday. can't wait 🙄"` is labeled `negative` but the rule model returns `positive`, because `great` scores +1 and 🙄 (the sarcasm signal) matches nothing. A keyword model has no mechanism to invert surface sentiment.
- **`mixed` is structurally unreachable in the rule model.** `predict_label` can only return positive/negative/neutral, so `"Feeling tired but kind of hopeful"` (true `mixed`) can never be predicted correctly — `tired` (−1) and `hopeful` (+1) even cancel to a score of 0 → neutral. Three of the dataset errors are this single cause.
- **Emoji and slang sentiment ignored.** `"it's fine. everything is fine. 💀"` (sarcastic distress, true `negative`) → neutral, because `fine` and 💀 are in no word list. `"passed my exam!! screaming rn 😭💅"` (true `positive`) → neutral for the same reason.
- **Over-broad negation.** Slang `no` in `"no cap"` and `no complaints` misfires as a grammatical negator, actively producing *wrong-direction* labels rather than just misses.
- **Small dataset / inflated ML metric.** The 1.00 ML score is training accuracy on 18 examples; it does not reflect real-world performance. No held-out test set exists.
- **Short text only.** Both models were only tested on one-line posts; behavior on longer text is unknown.

## 7. Bias and Scope

**Who this is optimized for:** English-speaking users fluent in **current internet/social-media slang** — `lowkey`, `no cap`, `ngl`, `fire`, `hyped` — and in emoji conventions (😭 as joy, 💀 as deadpan, 🙄 as sarcasm). The dataset reflects one specific cultural and generational register.

**Who it might misinterpret:**
- **Speakers of other English dialects or non-native speakers**, whose phrasing isn't in the word lists and wasn't in the training posts.
- **Users who use emojis differently** — 😭 can mean grief for one person and delight for another; the ML model learned only this dataset's convention and the rule model ignores emojis entirely.
- **Older or more formal writers**, whose sincere language could be misread by a model tuned on irony and slang.
- **Sarcasm across cultures** varies widely; a model that already fails on sarcasm will fail unevenly across groups.

Because the labels encode *my* interpretation of ambiguous posts (the edge cases in §2), the model also inherits my personal reading of tone — another labeler might disagree, and the model would then be "biased" toward my judgment.

## 8. Rule-Based vs. ML Comparison

- **Did the learned model behave differently?** Yes. It produced `mixed` labels the rule model cannot, and it "solved" sarcasm and slang posts the rule model got wrong.
- **Did it fix failures or introduce new ones?** On paper it fixed every rule-based failure (0.39 → 1.00). But this is illusory: it didn't *understand* sarcasm, it **memorized** the labeled posts. The new, hidden failure is **overfitting** — a perfect training score that says nothing about unseen text.
- **How sensitive was it to my labels?** Very. The ML model is shaped entirely by the labels I chose — it would happily learn a *mislabeled* post and still report 1.00. The rule model, by contrast, ignores my labels at "training" time and only changes when I edit its logic/vocabulary, so its accuracy honestly rose and fell with the data's difficulty. **The ML model is more sensitive to the data; the rule model's score is the more trustworthy signal**, because the ML metric is inflated by evaluating on training data.

## 9. Ideas for Improvement

- **Add a real train/test split or leave-one-out cross-validation** so the ML accuracy reflects generalization, not memorization (the single most important fix).
- **Collect much more labeled data**, balanced across the four classes and multiple labelers, to reduce personal bias.
- **Emoji/slang lexicon** for the rule model (e.g. 💀🙄 → negative, 😂🔥😭 → positive) and smarter negation that ignores slang idioms like "no cap."
- **Add a `mixed` branch** to the rule model by tracking positive and negative hit counts separately instead of a single signed score.
- **Use TF-IDF** instead of raw counts, or a small transformer, to capture context and sarcasm the bag-of-words model cannot.
