# ============================================================
#  COMPETITION-GRADE TWEET SENTIMENT ANALYSIS PIPELINE
#  Target: Positive / Neutral / Negative  (polarity 4 / 2 / 0)
#  Strategy:
#    Tier-1  – Classical ML ensemble  (fast, strong baseline)
#    Tier-2  – Fine-tuned BERTweet    (SOTA for tweets)
#    Tier-3  – Meta-ensemble          (blend both tiers)
# ============================================================

# ── INSTALL DEPENDENCIES (run once) ─────────────────────────
# pip install transformers datasets accelerate torch
# pip install contractions emoji scikit-learn xgboost lightgbm
# pip install wordcloud matplotlib seaborn pandas numpy tqdm

# ============================================================
# 0.  IMPORTS & CONFIGURATION
# ============================================================
import os, re, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

# NLP utilities
import nltk
import contractions
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# Scikit-learn
from sklearn.model_selection        import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model           import LogisticRegression
from sklearn.svm                    import LinearSVC
from sklearn.calibration            import CalibratedClassifierCV
from sklearn.ensemble               import VotingClassifier, StackingClassifier
from sklearn.metrics                import (accuracy_score, classification_report,
                                            confusion_matrix, f1_score,
                                            precision_recall_fscore_support)
from sklearn.preprocessing          import LabelEncoder
from sklearn.pipeline               import Pipeline

# Gradient boosting
import lightgbm as lgb
from xgboost import XGBClassifier

# PyTorch / HuggingFace
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

warnings.filterwarnings('ignore')

# ── Reproducibility ─────────────────────────────────────────
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

# ── Device ───────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Global label map ─────────────────────────────────────────
LABEL_MAP     = {0: 0, 2: 1, 4: 2}            # polarity → 0-based index
INV_LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}
CLASS_NAMES   = ['negative', 'neutral', 'positive']


# ============================================================
# 1.  LOAD & VALIDATE DATA
# ============================================================
def load_data(path: str = 'tweets.csv') -> pd.DataFrame:
    """Load raw CSV; handle both 6-column Sentiment140 and any other layout."""
    cols = ['polarity', 'id', 'date', 'query', 'user', 'text']
    try:
        df = pd.read_csv(path, header=None, names=cols, encoding='latin-1')
    except Exception:
        df = pd.read_csv(path, encoding='latin-1')

    # Basic sanity checks
    assert 'polarity' in df.columns, "Missing 'polarity' column."
    assert 'text'     in df.columns, "Missing 'text' column."

    df['polarity'] = df['polarity'].astype(int)
    df['label']    = df['polarity'].map(LABEL_MAP)
    df             = df.dropna(subset=['text', 'label']).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows.  Columns: {list(df.columns)}")
    print(df['label'].value_counts().rename(INV_LABEL_MAP))
    return df


# ============================================================
# 2.  EXPLORATORY DATA ANALYSIS
# ============================================================
def run_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2a. Class distribution
    counts = df['label'].value_counts().sort_index()
    axes[0].bar([INV_LABEL_MAP[i] for i in counts.index], counts.values,
                color=['#e74c3c','#95a5a6','#2ecc71'])
    axes[0].set_title('Class Distribution'); axes[0].set_ylabel('Count')

    # 2b. Tweet length distribution
    df['char_len'] = df['text'].str.len()
    for lbl in [0,1,2]:
        axes[1].hist(df[df['label']==lbl]['char_len'], bins=40,
                     alpha=0.6, label=INV_LABEL_MAP[lbl])
    axes[1].set_title('Tweet Length (chars)'); axes[1].legend()

    # 2c. Top-20 words per class (raw)
    from sklearn.feature_extraction.text import CountVectorizer
    for lbl, color in zip([0,2], ['#e74c3c','#2ecc71']):
        corpus = df[df['label']==lbl]['text'].fillna('').tolist()
        cv = CountVectorizer(max_features=20, stop_words='english')
        cv.fit(corpus)
        freqs = sorted(cv.vocabulary_.items(), key=lambda x: x[1])
        print(f"\nTop words [{INV_LABEL_MAP[lbl]}]: {[w for w,_ in freqs[:10]]}")

    plt.tight_layout(); plt.savefig('eda_overview.png', dpi=120); plt.show()
    print("EDA saved → eda_overview.png")


# ============================================================
# 3.  TEXT PREPROCESSING
# ============================================================
SLANG_DICT = {
    "u":"you","r":"are","ur":"your","n":"and","luv":"love","gr8":"great",
    "b4":"before","2day":"today","lol":"laughing out loud","omg":"oh my god",
    "tbh":"to be honest","imo":"in my opinion","ngl":"not gonna lie",
    "idk":"i don't know","brb":"be right back","btw":"by the way",
    "thx":"thanks","ty":"thank you","pls":"please","plz":"please",
    "smh":"shaking my head","irl":"in real life","fyi":"for your information"
}

_stop   = set(stopwords.words('english')) - {'not','no','never','very','but'}
_lemma  = WordNetLemmatizer()

def preprocess(text: str, keep_emoji_text: bool = True) -> str:
    """
    Full tweet cleaning pipeline:
      1. Lower-case
      2. Expand contractions   (can't → cannot)
      3. Convert emojis to text (❤ → heart)
      4. Remove URLs / mentions
      5. Expand slang
      6. Remove punctuation & numbers
      7. Lemmatise & strip stopwords
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = contractions.fix(text)

    if keep_emoji_text:
        text = emoji.demojize(text, delimiters=(' ', ' '))  # ❤ → heart

    # Remove URLs, mentions, RT prefix
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text)

    # Expand slang
    tokens = text.split()
    tokens = [SLANG_DICT.get(t, t) for t in tokens]
    text   = ' '.join(tokens)

    # Keep hashtag words (strip the #)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Lemmatise and remove stopwords
    tokens = [_lemma.lemmatize(t) for t in text.split()
              if t not in _stop and len(t) > 1]
    return ' '.join(tokens)


# ============================================================
# 4.  FEATURE ENGINEERING  (for classical ML tier)
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hand-crafted meta-features."""
    df = df.copy()
    df['char_count']       = df['text'].str.len()
    df['word_count']       = df['text'].str.split().str.len()
    df['url_count']        = df['text'].str.count(r'http\S+')
    df['mention_count']    = df['text'].str.count(r'@\w+')
    df['hashtag_count']    = df['text'].str.count(r'#\w+')
    df['exclamation_count']= df['text'].str.count(r'!')
    df['question_count']   = df['text'].str.count(r'\?')
    df['uppercase_ratio']  = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x)+1))
    df['emoji_count']      = df['text'].apply(
        lambda x: sum(1 for c in x if c in emoji.EMOJI_DATA))
    return df

META_FEATURES = ['char_count','word_count','url_count','mention_count',
                 'hashtag_count','exclamation_count','question_count',
                 'uppercase_ratio','emoji_count']


# ============================================================
# 5.  CLASSICAL ML TIER
# ============================================================
def build_classical_tier(X_train_tfidf, X_val_tfidf, X_test_tfidf,
                          meta_train, meta_val, meta_test,
                          y_train, y_val, y_test):
    """
    Train LR + SVM + LightGBM on TF-IDF + meta features.
    Return: best model, test predictions, test probabilities.
    """
    from scipy.sparse import hstack, csr_matrix

    # Combine TF-IDF with meta features
    X_tr  = hstack([X_train_tfidf, csr_matrix(meta_train)])
    X_vl  = hstack([X_val_tfidf,   csr_matrix(meta_val)])
    X_te  = hstack([X_test_tfidf,  csr_matrix(meta_test)])

    # ── Logistic Regression ──────────────────────────────────
    lr = LogisticRegression(C=5.0, max_iter=2000, solver='saga',
                             class_weight='balanced', random_state=SEED, n_jobs=-1)
    lr.fit(X_tr, y_train)
    lr_val_f1 = f1_score(y_val, lr.predict(X_vl), average='macro')
    print(f"  LR   val macro-F1 : {lr_val_f1:.4f}")

    # ── SVM (calibrated for probabilities) ──────────────────
    svm = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000, class_weight='balanced',
                      random_state=SEED), cv=3)
    svm.fit(X_tr, y_train)
    svm_val_f1 = f1_score(y_val, svm.predict(X_vl), average='macro')
    print(f"  SVM  val macro-F1 : {svm_val_f1:.4f}")

    # ── LightGBM ─────────────────────────────────────────────
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1)
    lgbm.fit(X_tr, y_train,
             eval_set=[(X_vl, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=-1)])
    lgbm_val_f1 = f1_score(y_val, lgbm.predict(X_vl), average='macro')
    print(f"  LGBM val macro-F1 : {lgbm_val_f1:.4f}")

    # ── Soft-vote ensemble ───────────────────────────────────
    probs_val = (lr.predict_proba(X_vl) +
                 svm.predict_proba(X_vl) +
                 lgbm.predict_proba(X_vl)) / 3

    probs_test = (lr.predict_proba(X_te) +
                  svm.predict_proba(X_te) +
                  lgbm.predict_proba(X_te)) / 3

    ens_val_f1 = f1_score(y_val, probs_val.argmax(1), average='macro')
    print(f"  ENSEMBLE val macro-F1 : {ens_val_f1:.4f}")

    return probs_test, probs_val   # shape (N, 3)


# ============================================================
# 6.  BERTTWEET TIER  (fine-tuned transformer)
# ============================================================
BERT_MODEL   = "vinai/bertweet-base"   # twitter-specific RoBERTa
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 4
LR_BERT      = 2e-5
WARMUP_RATIO = 0.1

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt')
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_bertweet(df_train, df_val, df_test):
    """Fine-tune BERTweet; return test and validation probabilities."""
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    BERT_MODEL, num_labels=3, ignore_mismatched_sizes=True)
    model.to(DEVICE)

    # Use RAW text for BERT (it handles its own tokenisation better)
    raw_train = df_train['text'].fillna('').tolist()
    raw_val   = df_val['text'].fillna('').tolist()
    raw_test  = df_test['text'].fillna('').tolist()

    train_ds = TweetDataset(raw_train, df_train['label'].tolist(), tokenizer)
    val_ds   = TweetDataset(raw_val,   df_val['label'].tolist(),   tokenizer)
    test_ds  = TweetDataset(raw_test,  df_test['label'].tolist(),  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    optimizer  = AdamW(model.parameters(), lr=LR_BERT, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler  = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(total_steps * WARMUP_RATIO),
                    num_training_steps=total_steps)

    best_val_f1 = 0
    best_state  = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ───────────────────────────────────────────
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            train_loss += loss.item()

        # ── Validate ────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits
                preds  = logits.argmax(-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"  Epoch {epoch}  avg_loss={train_loss/len(train_loader):.4f}  val_macro-F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Inference on best checkpoint ────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    def get_probs(loader):
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    test_probs = get_probs(test_loader)
    val_probs  = get_probs(val_loader)
    print(f"\nBERTweet best val macro-F1: {best_val_f1:.4f}")
    return test_probs, val_probs


# ============================================================
# 7.  META-ENSEMBLE  (learned blending)
# ============================================================
def meta_ensemble(classical_val_probs, bert_val_probs,
                  classical_test_probs, bert_test_probs,
                  y_val):
    """
    Learn optimal blend weight α on the validation set:
      final_probs = α * BERT + (1-α) * Classical
    Then apply to test set.
    """
    best_alpha, best_f1 = 0.5, 0
    for alpha in np.arange(0.0, 1.05, 0.05):
        blended  = alpha * bert_val_probs + (1 - alpha) * classical_val_probs
        f1       = f1_score(y_val, blended.argmax(1), average='macro')
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha

    print(f"\nOptimal α (BERT weight): {best_alpha:.2f}  val macro-F1: {best_f1:.4f}")
    final_probs = best_alpha * bert_test_probs + (1 - best_alpha) * classical_test_probs
    return final_probs.argmax(1), final_probs


# ============================================================
# 8.  EVALUATION & VISUALISATION
# ============================================================
def full_evaluation(y_true, y_pred, y_probs=None, title='Test Set'):
    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    p_wt,  r_wt,  f1_wt,  _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')

    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro  Precision  : {p_mac:.4f}")
    print(f"  Macro  Recall     : {r_mac:.4f}")
    print(f"  Macro  F1         : {f1_mac:.4f}")
    print(f"  Weighted Precision: {p_wt:.4f}")
    print(f"  Weighted Recall   : {r_wt:.4f}")
    print(f"  Weighted F1       : {f1_wt:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title(f'Confusion Matrix – {title}')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    # Per-class F1 bar
    p_pc, r_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0,1,2])
    x = np.arange(3)
    w = 0.25
    axes[1].bar(x-w, p_pc,  w, label='Precision', color='#3498db')
    axes[1].bar(x,   r_pc,  w, label='Recall',    color='#2ecc71')
    axes[1].bar(x+w, f1_pc, w, label='F1',        color='#e74c3c')
    axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylim(0, 1.05); axes[1].legend()
    axes[1].set_title('Per-class Metrics')

    plt.tight_layout()
    plt.savefig(f'eval_{title.replace(" ","_")}.png', dpi=120)
    plt.show()

    return {'accuracy': acc, 'macro_f1': f1_mac, 'weighted_f1': f1_wt}


# ============================================================
# 9.  MAIN PIPELINE
# ============================================================
def main(csv_path='tweets.csv', use_bert=True):
    # ── 9.1  Load ────────────────────────────────────────────
    df = load_data(csv_path)
    run_eda(df)

    # ── 9.2  Add meta features (on raw text) ─────────────────
    df = add_features(df)

    # ── 9.3  Preprocess text ─────────────────────────────────
    print("\nCleaning tweets …")
    df['clean_text'] = df['text'].apply(preprocess)

    # ── 9.4  Stratified splits  70 / 15 / 15 ─────────────────
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df['label'])
    df_val, df_test   = train_test_split(
        df_temp, test_size=0.50, random_state=SEED, stratify=df_temp['label'])

    print(f"\nSplit sizes → train:{len(df_train):,}  val:{len(df_val):,}  test:{len(df_test):,}")

    y_train = df_train['label'].values
    y_val   = df_val['label'].values
    y_test  = df_test['label'].values

    # ── 9.5  TF-IDF (fit ONLY on train) ──────────────────────
    print("\nFitting TF-IDF vectoriser …")
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        analyzer='word',
        strip_accents='unicode'
    )
    X_train_tfidf = tfidf.fit_transform(df_train['clean_text'])
    X_val_tfidf   = tfidf.transform(df_val['clean_text'])
    X_test_tfidf  = tfidf.transform(df_test['clean_text'])

    # ── 9.6  Classical tier ───────────────────────────────────
    print("\n[Tier-1] Classical ML Ensemble …")
    classical_test_probs, classical_val_probs = build_classical_tier(
        X_train_tfidf, X_val_tfidf, X_test_tfidf,
        df_train[META_FEATURES].values,
        df_val[META_FEATURES].values,
        df_test[META_FEATURES].values,
        y_train, y_val, y_test
    )

    # Evaluate classical tier standalone
    print("\n── Classical tier standalone ──")
    full_evaluation(y_test, classical_test_probs.argmax(1),
                    classical_test_probs, title='Classical Ensemble')

    # ── 9.7  BERTweet tier ───────────────────────────────────
    if use_bert:
        print("\n[Tier-2] Fine-tuning BERTweet …")
        bert_test_probs, bert_val_probs = train_bertweet(df_train, df_val, df_test)

        # Evaluate BERT standalone
        print("\n── BERTweet standalone ──")
        full_evaluation(y_test, bert_test_probs.argmax(1),
                        bert_test_probs, title='BERTweet')

        # ── 9.8  Meta-ensemble ───────────────────────────────
        print("\n[Tier-3] Meta-ensemble …")
        final_preds, final_probs = meta_ensemble(
            classical_val_probs, bert_val_probs,
            classical_test_probs, bert_test_probs,
            y_val
        )
    else:
        final_preds  = classical_test_probs.argmax(1)
        final_probs  = classical_test_probs

    # ── 9.9  Final evaluation ────────────────────────────────
    print("\n[FINAL] Meta-ensemble on TEST SET")
    results = full_evaluation(y_test, final_preds, final_probs,
                               title='Final Ensemble (Test Set)')

    # ── 9.10  Save predictions CSV ───────────────────────────
    out = df_test[['id','text','label']].copy()
    out['predicted_label']     = final_preds
    out['predicted_sentiment'] = out['predicted_label'].map(INV_LABEL_MAP)
    out['prob_negative']       = final_probs[:, 0]
    out['prob_neutral']        = final_probs[:, 1]
    out['prob_positive']       = final_probs[:, 2]
    out.to_csv('predictions.csv', index=False)
    print("\nPredictions saved → predictions.csv")

    return results


# ============================================================
# 10.  ENTRY POINT
# ============================================================
if __name__ == '__main__':
    # Set use_bert=False to run only the classical tier (faster, no GPU needed).
    # Set use_bert=True  for the full SOTA pipeline (requires GPU for speed).
    main(csv_path='tweets.csv', use_bert=True)


# ============================================================
# METHODOLOGY NOTES
# ============================================================
"""
WHY THESE CHOICES?
──────────────────
Preprocessing
  • Contractions expanded (can't→cannot) avoids splitting sentiment signal.
  • Emojis converted to text (❤→heart) — emojis are strong sentiment carriers.
  • Slang dictionary normalises informal Twitter language.
  • Stopwords list keeps negation words ('not','no','never') because they
    completely reverse sentiment meaning.

Feature Engineering
  • Meta-features (emoji count, uppercase ratio, exclamations) capture
    stylistic sentiment cues that TF-IDF misses.

TF-IDF
  • tri-grams (1,3) capture short phrases like "not good at all".
  • sublinear_tf dampens the dominance of very frequent terms.
  • 50k features keeps vocabulary rich without memory issues.

Classical Tier  (LR + SVM + LightGBM soft-vote)
  • Each model has different inductive biases; soft-voting averages out
    individual errors and produces calibrated probabilities.
  • Class-weighted loss handles potential imbalance.
  • CalibratedClassifierCV on LinearSVC produces proper probabilities for blending.

BERTweet (vinai/bertweet-base)
  • Pre-trained on 850M tweets → understands hashtags, mentions, abbreviations
    natively without normalisation.
  • State-of-the-art for tweet-level NLP benchmarks (TweetEval).
  • Early stopping via checkpoint saving prevents over-fitting.
  • Linear warmup + weight decay = standard fine-tuning best practice.

Meta-Ensemble
  • Learns the optimal BERT vs. Classical blend on held-out validation data.
  • Simple linear blend is interpretable and avoids meta-overfitting.

Evaluation
  • Macro-F1 is the primary metric: treats all classes equally regardless of
    support, crucial when classes are imbalanced (neutral is rare in Sentiment140).
  • We report accuracy, precision, recall, weighted-F1 for completeness.

Potential Further Improvements (if time allows)
  • Use RoBERTa-large-tweet or DeBERTa-v3 for higher ceiling.
  • Pseudo-labelling unlabelled tweets to augment training data.
  • Multi-task learning (sentiment + emotion simultaneously).
  • Data augmentation: back-translation or synonym substitution.
"""
