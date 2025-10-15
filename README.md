<div align="center" style="padding:22px 0 10px;">
  <h1 style="margin:0 0 6px;font-size:34px;line-height:1.15;">
    Detecting Multiclass Depressive Tweets with BERT
  </h1>
  <p style="margin:0;color:#6a737d;">
    Contextual vs Non-Contextual Word Embeddings (BERT vs FastText vs Word2Vec)
  </p>

  <p style="margin-top:10px;">
    <a href="https://img.shields.io/badge/Python-3.8%2B-informational?logo=python" target="_blank"><img src="https://img.shields.io/badge/Python-3.8%2B-informational?logo=python" alt="Python"></a>
    <a href="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" alt="TensorFlow"></a>
    <a href="https://img.shields.io/badge/Transformers-HF-6C5CE7?logo=huggingface" target="_blank"><img src="https://img.shields.io/badge/Transformers-HF-6C5CE7?logo=huggingface" alt="HF Transformers"></a>
    <a href="https://img.shields.io/badge/License-MIT-green" target="_blank"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  </p>

  <div style="margin-top:12px; text-align:left; max-width:900px;">
    <blockquote style="border-left:4px solid #e1e4e8; margin:0; padding:8px 12px; color:#586069;">
      <b>Thesis title:</b> “Kontekstual dan Non-Kontekstual Word Embedding dalam Mendeteksi Tweet Multiclass Bernuansa Depresif di Twitter Menggunakan LSTM”<br/>
      <b>Author:</b> Arif Farhan Bukhori (18/430254/PA/18767) — Universitas Gadjah Mada (FMIPA – Ilmu Komputer)
    </blockquote>
  </div>
</div>

<hr style="border:none;border-top:1px solid #eaecef;"/>

<h2 id="abstract" style="margin-top:24px;">Abstract</h2>
<p>
We compare <b>BERT (contextual)</b> against <b>FastText</b> and <b>Word2Vec</b> (non-contextual) as embeddings for an <b>LSTM</b> classifier that labels tweets into four classes: <i>depressive</i>, <i>negative</i>, <i>neutral</i>, <i>positive</i>. Using pre-trained embeddings and a consistent LSTM backbone, the <b>BERT + LSTM</b> pipeline delivers the strongest performance with <b>≈95% macro/weighted F1</b> on a combined dataset (&gt;9k tweets). Word2Vec/FastText are faster to train but lag in accuracy/F1.
</p>

<h2 id="highlights" style="margin-top:24px;">Highlights</h2>
<ul>
  <li>4-way classification: depressive / negative / neutral / positive</li>
  <li>Embeddings compared: <b>Word2Vec</b>, <b>FastText</b>, <b>BERT-Base</b></li>
  <li>Backbone: <b>TensorFlow/Keras LSTM</b> with average pooling, ReLU dense, softmax</li>
  <li>Best config: <b>batch_size=16</b>, <b>dropout=0.2</b>, early stopping on val_acc</li>
  <li>Cross-validation: 5-fold per scenario; 80/20 train/test split for final eval</li>
</ul>

<h2 id="dataset" style="margin-top:24px;">Dataset</h2>
<p>
Combined public tweet corpora with labels:
</p>
<ul>
  <li><b>Depressive</b> tweets: Yang (2020) ≈ 2308 + Romero (2019) ≈ 2314 → <b>4622</b></li>
  <li><b>Sentiment</b> tweets (Yadav, 2018): Negative 1870, Positive 1486, Neutral 1117</li>
</ul>
<p>Total: <b>≈9132 tweets</b>, balanced through evaluation with weighted metrics.</p>

<h2 id="results" style="margin-top:24px;">Results (Test Set)</h2>
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Accuracy</th>
      <th>Time / epoch*</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>BERT + LSTM</b></td>
      <td>0.9525</td>
      <td>0.9521</td>
      <td><b>0.9520</b></td>
      <td>0.9521</td>
      <td>~442 s</td>
    </tr>
    <tr>
      <td>FastText + LSTM</td>
      <td>0.9037</td>
      <td>0.9019</td>
      <td>0.9024</td>
      <td>0.9019</td>
      <td>~8 s</td>
    </tr>
    <tr>
      <td>Word2Vec + LSTM</td>
      <td>0.9081</td>
      <td>0.9081</td>
      <td>0.9079</td>
      <td>0.9081</td>
      <td>~8 s</td>
    </tr>
  </tbody>
</table>
</div>
<p style="color:#6a737d; font-size:13px; margin-top:6px;">*Approximate training on Colab V100 (Pro+) as reported in the thesis.</p>

<h2 id="method" style="margin-top:24px;">Method Overview</h2>
<ol>
  <li><b>Preprocessing</b>: contractions expansion, mention/URL removal, punctuation/whitespace cleanup, case-folding, tokenization, padding to 140 tokens.</li>
  <li><b>Embeddings</b>:
    <ul>
      <li><b>Word2Vec</b>: Google News (300-d).</li>
      <li><b>FastText</b>: pre-trained (300-d).</li>
      <li><b>BERT-Base</b>: WordPiece tokenizer; 768-d sequence features.</li>
    </ul>
  </li>
  <li><b>Model</b>: Embedding → LSTM (return sequences) → AvgPooling → Dense(150, ReLU) → Dense(4, Softmax).</li>
  <li><b>Training</b>: Adam (lr=1e-3), categorical cross-entropy, batch_size∈{16,32}, dropout∈{0,0.2}, early stopping (patience=3).</li>
  <li><b>Evaluation</b>: Accuracy, Precision, Recall, F1 (weighted & macro), Confusion Matrix.</li>
</ol>

<h2 id="quickstart" style="margin-top:24px;">Quick Start</h2>
<details>
  <summary><b>Run in Google Colab</b></summary>
  <ol>
    <li>Open Colab and enable GPU.</li>
    <li>Install deps: <code>pip install tensorflow transformers gensim fasttext scikit-learn matplotlib pandas numpy tqdm</code></li>
    <li>Download/prep datasets, then run the notebooks:
      <ul>
        <li><i>01_preprocess.ipynb</i> – cleaning & EDA</li>
        <li><i>02_word2vec_lstm.ipynb</i>, <i>03_fasttext_lstm.ipynb</i>, <i>04_bert_lstm.ipynb</i></li>
        <li><i>05_eval_compare.ipynb</i> – metrics + confusion matrix</li>
      </ul>
    </li>
  </ol>
</details>

<details>
  <summary><b>Local (Python 3.8+)</b></summary>
  <pre style="background:#f6f8fa;padding:10px;border-radius:6px;overflow:auto;"># 1) Create env & install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Put datasets under data/

# data/

# depressive\_yang.csv

# depressive\_romero.csv

# sentiment\_yadav.csv

# 3) Run a pipeline (example: BERT)

python src/run\_bert\_lstm.py&#x20;
\--train data/train.csv --test data/test.csv&#x20;
\--batch\_size 16 --dropout 0.2 --epochs 20 --lr 1e-3

# 4) View metrics & confusion matrix in outputs/

</pre>
</details>

<h2 id="repo-structure" style="margin-top:24px;">Suggested Repo Structure</h2>
<pre style="background:#f6f8fa;padding:10px;border-radius:6px;overflow:auto;">.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # original CSVs
│   └── processed/    # cleaned, tokenized splits
├── src/
│   ├── preprocess.py
│   ├── utils.py
│   ├── run_word2vec_lstm.py
│   ├── run_fasttext_lstm.py
│   └── run_bert_lstm.py
├── notebooks/
│   ├── 01_preprocess_and_eda.ipynb
│   ├── 02_word2vec_lstm.ipynb
│   ├── 03_fasttext_lstm.ipynb
│   └── 04_bert_lstm.ipynb
└── outputs/
    ├── models/
    ├── metrics/
    └── figures/
</pre>

<h2 id="ethics" style="margin-top:24px;">Ethics & Responsible Use</h2>
<ul>
  <li>This repository is for <b>research and educational</b> purposes.</li>
  <li>Never use model outputs to make clinical decisions. Consult licensed professionals.</li>
  <li>Comply with the source platforms’ ToS and regional data protection laws.</li>
</ul>

<h2 id="citation" style="margin-top:24px;">Citation</h2>
<p>If you use this work, please cite:</p>
<pre style="background:#f6f8fa;padding:10px;border-radius:6px;overflow:auto;">@thesis{Bukhori2022DepressiveTweets,
  author    = {Arif Farhan Bukhori},
  title     = {Contextual and Non-Contextual Word Embedding in Detecting Depressive Multiclass Tweets on Twitter Using LSTM},
  school    = {Universitas Gadjah Mada},
  year      = {2022}
}
</pre>

<h2 id="ack" style="margin-top:24px;">Acknowledgments</h2>
<p>Advisors and mentors at UGM FMIPA; open-source communities behind TensorFlow, Keras, Hugging Face, Gensim, scikit-learn; and dataset contributors (Yang, Romero, Yadav).</p>

<h2 id="license" style="margin-top:24px;">License</h2>
<p>MIT — see <code>LICENSE</code>.</p>

<hr style="border:none;border-top:1px solid #eaecef;"/>
<p style="font-size:13px;color:#6a737d;">
Maintainer: <a href="https://github.com/riffarhan" target="_blank">@riffarhan</a> • PRs and issues welcome.
</p>


