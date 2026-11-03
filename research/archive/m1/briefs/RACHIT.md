# Rachit Kumar — read this before tomorrow

**Roll no:** 230128 · **You present:** slide 5, *Ensembles, interpretability & transformers* · **Your time:** ~2 minutes

> This is the only file you need. You do not have to open the slides, read any paper,
> or look anything up. Read this once (~15 min), and you can walk in and talk.
> Everything is explained from scratch — no jargon is used before it is defined.

---

## 0 · If you only remember four sentences

1. We are reviewing 12 papers on **detecting pneumonia in children's chest X-rays**
   using AI, on a public dataset of 5,856 images from one hospital in Guangzhou, China.
2. Over six years, reported accuracy on that dataset climbed from **92.8% to 98%** — but
   the test set is only **624 images** and the validation set is **16 images**.
3. **Our research gap:** the papers using this dataset never check whether their models
   actually work — no external hospital, no error bars, no calibration, and their
   explanation heatmaps are shown but never scored. The papers that *do* run those checks
   all use different datasets. **The two literatures never touch.**
4. **You present the three ways people pushed past 96%** — ensemble five networks,
   visualise what one network sees, or replace convolution with a Vision Transformer.

---

## 1 · What is this project, actually?

It is a **research-reproduction project**. We are not inventing something from nothing.
The assignment has six steps, and the whole course is graded across five milestones:

| # | Step | What it means |
|---|------|---------------|
| 1 | **Literature survey** | Read 10–20 recent papers on our topic. ← **this is tomorrow (M1)** |
| 2 | Pick a base paper | Choose ONE paper with public code + downloadable data |
| 3 | Reproduce | Re-run their method, land near their reported number |
| 4 | Hypothesis | Propose ONE change of our own, and **predict the result before coding** |
| 5 | Implement & test | Make the change, run ablations, measure what moved |
| 6 | Analyse & write | Failure analysis + write it up like a real IEEE conference paper |

**Tomorrow is step 1 only.** We present, in 10 minutes, what we read and what we found.

There is one extra instruction that matters more than the rest. The exact words were:

> *"Bonus points if you can find a problem/shortcoming still present in all of them
> which none of the papers have solved (this is called a research gap)."*

**We found one.** That is the strongest part of our presentation, and every one of the
four of us should understand it, because that is what she is most likely to ask about.

Our chosen topic (already submitted): **Pneumonia detection from chest X-rays,
using the Kermany CXR dataset.**

---

## 2 · The medical problem, in plain words

**Pneumonia** is an infection that inflames the tiny air sacs in the lungs (alveoli).
They fill with fluid or pus instead of air. It is one of the largest causes of death
in children under five worldwide.

**How it is diagnosed:** a chest X-ray. Healthy lungs are mostly air, and air lets
X-rays pass straight through, so healthy lungs look **dark** on the film. When the air
sacs fill with fluid, that region blocks more X-rays and shows up as a **white or hazy
patch** — radiologists call this an *opacity* or *consolidation*.

**So why is this hard?**

1. The patch is often faint, and can hide behind the heart or the ribs.
2. Other conditions look similar on film.
3. In children it is even harder — smaller anatomy, and kids move during the scan.
4. Most importantly, **there are not enough radiologists.** In many parts of the world
   an X-ray gets taken and then waits hours or days for someone qualified to read it.
   That delay is where children die.

So the goal is **not** to replace a doctor. The goal is a **triage tool** — something
that flags the urgent films so a human looks at those first.

**One more distinction that comes up constantly:** pneumonia can be **bacterial** or
**viral**. This matters enormously because bacterial pneumonia needs antibiotics and
viral pneumonia does not. Telling them apart on an X-ray is much harder than just
detecting that *something* is wrong — and you will see every paper score worse on it.

---

## 3 · Where computer vision comes in

The task we are doing is **image classification**: input one X-ray, output one label.

Everything in this field is built from four ideas. Learn these four and you can follow
any of the twelve papers.

### 3.1 · CNN — Convolutional Neural Network

The workhorse. Think of it as looking at the image through a small sliding window:

- **Early layers** detect very simple things — edges, corners, light-to-dark transitions.
- **Middle layers** combine those into textures and shapes — rib curves, hazy regions.
- **Late layers** combine *those* into concepts — "a cloudy white area in the lower left lung."
- **Final layer** collapses everything into one number: the probability of pneumonia.

Crucially, **nobody tells it what to look for.** You show it thousands of labelled images,
it makes guesses, you tell it how wrong it was, and it nudges its internal numbers
(*weights*) to be less wrong next time. Repeat millions of times. It works out its own
features. This is why the "what is it actually using?" question is so hard — and why
it is the heart of our project.

### 3.2 · Transfer learning

Training a CNN from zero needs millions of images. We have 5,856.

The trick: take a network **already trained on ImageNet** — 1.4 million everyday
photographs of cats, cars, chairs. It has already learned edges, curves, textures, and
how to combine them. Then keep those early layers ("**freeze**" them) and retrain only
the last few layers on X-rays. This is called **fine-tuning**.

You get a strong model from a few thousand images instead of a few million.
**Almost every paper we read does exactly this.** The differences between papers are
mostly just *which* pretrained network they grabbed.

The networks you will hear named — all of them are just different CNN designs:

| Name | The one thing to know |
|------|----------------------|
| **VGG16** | Old and simple: a plain deep stack of small 3×3 filters. Sturdy, heavy. |
| **Inception-V3** | Looks at several filter sizes in parallel at each layer, then merges. |
| **ResNet** | Adds *skip connections* so a layer can pass its input forward unchanged. This is what let networks get really deep without training collapsing. |
| **DenseNet** | Takes that further — every layer receives input from *all* previous layers. Best performer on this task in several papers. |
| **Xception** | Splits each convolution into two cheaper steps (space first, then channels). Efficient. |
| **ViT** (Vision Transformer) | Not a CNN at all — explained in §9 if it is on your slide. |

### 3.3 · Data augmentation

We only have 5,216 training images, and a big network can simply **memorise** them
(this is called **overfitting** — perfect on training data, useless on new data).

Fix: make fake-but-realistic extra copies. Rotate slightly, zoom, shift, flip, adjust
brightness. The lungs are still lungs, the label is still correct, but the network can
no longer memorise exact pixels — it has to learn the actual pattern.

### 3.4 · Grad-CAM — the "explanation" heatmap

A coloured heatmap laid over the X-ray, showing which pixels most influenced the answer.
Red = mattered a lot. Blue = did not matter.

It works by taking the last convolutional layer's feature maps and weighting them by how
much each one pushed the final "pneumonia" score up.

**This is the standard way papers claim "our model looks at the right place."**
Remember that phrasing — *claim*. Whether that claim survives being measured is a
central part of our story.

---

## 4 · Our dataset — know these numbers cold

This is the **Kermany chest X-ray dataset** (also on Kaggle as "Chest X-Ray Images
(Pneumonia)"). Released alongside the Kermany 2018 paper.

| Split | Images |
|-------|--------|
| Train | **5,216** |
| Test | **624** |
| Validation | **16** |
| **Total** | **5,856** |

Four facts that drive our entire argument:

1. **One hospital.** Guangzhou Women and Children's Medical Center, China.
   One site, one set of machines, one acquisition protocol, one population.
2. **Children aged 1–5 only.** Not adults. Not newborns.
3. **The classes are imbalanced: 4,273 pneumonia vs 1,583 normal**, roughly 2.7 : 1.
   Consequence: a model that stupidly answers "pneumonia" for **every single image**
   scores **62.5%** on the test set without learning anything. So "62.5%" is our floor,
   not 50%. Any accuracy number must be read against that.
4. **The validation split is 16 images.** Eight per class. Read that again.

Why (4) is a genuine scandal and not a nitpick:

> You need three splits. **Train** = what the model learns from. **Validation** = what you
> use to make decisions (when to stop training, which learning rate, which architecture).
> **Test** = touched once, at the very end, to report an honest number.
>
> With 16 validation images you cannot tell a 93% model from an 87% model — the
> difference is one image. So in practice everyone tunes against the **test** set and
> then reports on that same test set. That is called **selecting on the test set**, and it
> makes every published number optimistic by construction.

**And: 624 test images gives a 95% confidence interval of roughly ±1.5 percentage points.**
A "1% improvement" is about six images. Hold on to that number — it is the backbone of
our research gap.

---

## 5 · Jargon decoder

Anything you might need to say or defend. Skim now, come back if she asks.

| Term | Plain meaning |
|------|---------------|
| **Accuracy** | Fraction of all predictions that were correct. Misleading when classes are imbalanced (see the 62.5% floor above). |
| **Sensitivity / Recall** | Of the children who **actually have** pneumonia, what fraction did we catch? **This is the one that matters medically** — a missed sick child is the dangerous error. |
| **Specificity** | Of the **healthy** children, what fraction did we correctly clear? Low specificity = lots of false alarms. |
| **Precision** | Of the ones we *flagged* as pneumonia, what fraction really had it? |
| **F1 score** | A single number balancing precision and recall (their harmonic mean). Used when classes are imbalanced. |
| **AUC / AUROC** | The model outputs a score from 0 to 1. Accuracy forces you to pick a cutoff (say 0.5); AUC measures quality across **every possible cutoff**. Concretely: pick one random sick patient and one random healthy patient — AUC is the probability the model scores the sick one higher. **0.5 = coin flip, 1.0 = perfect.** |
| **Epoch** | One full pass through the training data. |
| **Overfitting** | Model memorises training data; great on train, bad on anything new. |
| **Ablation** | Deliberately remove/disable one component to see how much it was contributing. |
| **Confidence interval (CI)** | The range the true value probably sits in, given you only measured on a limited sample. Small test set → wide interval. |
| **Calibration** | When the model says "90% confident," is it right 90% of the time? Neural networks are famously **overconfident** — they say 99% when they are right 80% of the time. |
| **ECE** | *Expected Calibration Error.* One number summarising that mismatch. Lower is better. Zero papers we read report it. |
| **Segmentation** | Instead of one label for the whole image, label **every pixel** — e.g. "this pixel is lung, this one isn't." Output is a **mask**. |
| **U-Net** | The standard CNN design for segmentation. Shrinks the image down to understand context, then expands it back to full resolution to draw precise borders. |
| **External validation** | Testing on data from a **completely different hospital** than you trained on. The real test of whether a medical model works. |
| **Shortcut learning** | The model finds the **easiest** thing that predicts the label — even if it is medically meaningless. Explained fully in §7. |
| **Domain shift** | The new data looks different from the training data (different scanner, different hospital, different patients), so performance drops. |
| **Baseline** | The number you must beat. Ours is 62.5% (always-say-pneumonia). |

---

## 6 · The three papers the whole field is built on

*Samarth presents these in detail on slide 3. You need to recognise them, because
every other paper in our review is a descendant of one of them.*

### Kermany et al., 2018, *Cell* — the dataset + the recipe

Took **Inception-V3** pretrained on ImageNet, froze most of it, retrained the top on
medical images. Showed it works on pediatric chest X-rays: **92.8% accuracy, 93.2%
sensitivity, 90.1% specificity**, plus 90.7% on bacterial-vs-viral.

Two reasons it matters: it proved transfer learning works in medicine with small data,
and **they released the dataset publicly** — which is why it became the standard
benchmark that everyone else in our review uses. ~6,100 citations.

**This is also our chosen base paper for the reproduction step.**

### Rajpurkar et al., 2017 — **CheXNet** (Stanford)

Went big instead of clever: a **121-layer DenseNet** trained on 112,120 adult chest
X-rays, predicting 14 diseases at once. Claimed "radiologist-level" performance.

The number to actually remember: pneumonia **F1 = 0.435**, versus four radiologists
averaging **0.387**. It beat the humans — but **0.435 is a low score in absolute terms.**
Pneumonia is genuinely hard, even for people. Anyone quoting 98% should be made to
explain that gap.

This paper made **DenseNet + Grad-CAM** the default recipe for the entire field.
It is also one of very few that reported **bootstrap 95% confidence intervals** —
a detail that matters a lot in §8.

### Wang et al., 2017, CVPR — **ChestX-ray8** (NIH)

Released **108,948 X-rays from 32,717 patients**. They could not afford doctors to label
100,000 images, so they ran **NLP over the written radiology reports** to mine the labels
automatically. Cheap and scalable — but the labels are **noisy**, which caps how well
anything trained on it can score.

Second contribution, and the one that matters to us: they supplied **bounding boxes** for
a subset and **scored localization** — did the model's attention actually land on the
finding? — as a number, not a picture. That idea is the ancestor of what we propose to do.

---

## 7 · The story of the field, in one page

This is the arc of our whole presentation. If you understand this, you can improvise.

**2017–2018 — the tools arrive.** Public datasets appear (ChestX-ray8, Kermany). Transfer
learning is shown to work on small medical datasets. Grad-CAM becomes the standard way
to "explain" a prediction. Kermany reports **92.8%**.

**2018–2024 — the race.** Everyone downloads the same 5,856 images and swaps in a
different pretrained network. Reported accuracy climbs: 96.2 → 96.4 → 97.6 → **98.0**.
Ensembles appear. Vision Transformers appear. Every paper reports a higher number than
the last, on **the same 624 test images**.

**Meanwhile, in a completely separate literature — the reality check.** A different set
of researchers, publishing in *PLOS Medicine* and *Nature Machine Intelligence*, asked
whether models like these actually work. They found:

- Move the model to another hospital and performance **collapses** (Zech).
- The models are often keying on **things outside the lungs entirely** — equipment
  markers, image borders, how the patient was positioned (DeGrave). This is
  **shortcut learning**: a neural network finds the *easiest* feature that predicts the
  label, and if a shortcut exists in your data it will take it, because it is cheaper to
  learn than actual disease.
- And the Grad-CAM heatmaps everyone uses as proof? When you **score** them against a
  radiologist's own markings, they are **significantly worse than a human** (Saporta).

**And here is the punchline of our entire presentation:**

> **These two literatures never touch.**
>
> Every paper that uses our dataset reports one accuracy number and never checks any of
> this. Every paper that does the checking uses a *different* dataset.
> Nobody has run the reality check on this benchmark.

---

## 8 · THE RESEARCH GAP — the bonus-marks part

**Say it like this:**

> "Every paper on this benchmark reports a single accuracy number on the same 624
> single-centre images. Every paper that tested whether such models actually work used a
> different dataset. Nobody has run the reality check on the benchmark itself."

It breaks into five specific, checkable gaps. Not one of the twelve papers closes them:

| # | Gap | Why it matters |
|---|-----|----------------|
| **1** | **Selection on the test set** | The validation split is 16 images, so it cannot rank models. Tuning happens against the test set. Reported numbers are optimistic **by construction**. |
| **2** | **No error bars** | Zero of the nine benchmark papers report a confidence interval or run multiple random seeds. |
| **3** | **No external validation** | One hospital, one city, ages 1–5. Zech measured a **−0.12 AUC** drop off-site elsewhere. Nobody has measured it here. |
| **4** | **No calibration** | Not one paper reports ECE or a reliability curve. A triage tool needs a *trustworthy confidence*, not just a correct label. |
| **5** | **Heatmaps never scored** | Grad-CAM appears as figures in most papers. Saporta showed that actually scoring them **changes the verdict**. Nobody scores them here. |

### The statistics point — get this exactly right

There is a subtle version of gap 2 that is worth stating carefully, because getting it
half-right is worse than not saying it.

624 test images → the 95% confidence interval is about **±1.5 percentage points**.
Now lay the reported numbers against that window:

- Kermany 2018 → **92.8%**
- Rajaraman 2018 → **96.2%**
- Chouhan 2020 → **96.4%**
- Singh (ViT) 2024 → **97.6%**
- Rahman 2020 → **98.0%**

**The first jump is real.** 92.8 → 96.2 clears the ±1.5 window comfortably.

**Everything after that is not separable from noise.** 96.2, 96.4, 97.6 and 98.0 all sit
inside overlapping windows. So roughly **five years of published "improvement" cannot be
distinguished from random variation** — and because nobody reports a CI or a seed
variance, none of those authors could have known either way.

⚠️ **Do not say "all the gains are noise."** That is wrong and she may catch it.
Say: *"the first jump is real; every gain claimed since 2019 sits inside the noise window."*

---

## 9 · What we are actually going to build (milestones 2–5)

### Base paper: Kermany et al. 2018 (Inception-V3)

Chosen for **reproducibility, not performance**. It has public data, a clearly stated
architecture, and three clearly stated targets to land near: **92.8% / 93.2% / 90.1%**.
Small enough to train on a free GPU.
*(Fallback if reproduction stalls: Rahman 2020, DenseNet-201 — same data, more
implementations floating around.)*

### Our ONE change: train and test on **lung-cropped** X-rays

The assignment allows exactly one change of our own. Here it is:

1. Run a **pretrained lung segmentation model** (a U-Net trained on the public Montgomery
   and Shenzhen chest X-ray sets) over every image. It outputs a **mask** marking which
   pixels are lung.
2. **Crop to that region.** Everything the shortcut literature blames — laterality
   markers, collimation edges, image borders, burnt-in text, shoulders, positioning
   cues — is now *physically gone from the input*.
3. Train **two identical models**: one on full images, one on cropped. Same architecture,
   same hyperparameters, same seeds. **Only the input region changes.**

### Our prediction — submitted BEFORE we write any code

The rules say: *predict first, being wrong is fine, good reasoning is what counts.*

- **On the normal test set:** the two models will land within about **1 point** of each
  other. Cropping adds no information — the lungs already carry the real signal.
- **On a context-ablation stress test** — invert the mask, so we **black out the lungs**
  and keep only the surrounding context — the full-image model will **still score well
  above the 62.5% baseline**, because it can lean on shortcuts. The cropped model, having
  never seen that context, should **fall toward chance**.
- **That gap is the amount of shortcut, measured as a number.** Which is exactly what
  nobody has done on this dataset.
- Secondary predictions: the cropped model should be **better calibrated** (lower ECE)
  and have a **higher Grad-CAM hit rate inside the lung field**.

### And if the prediction is wrong?

Still a result, and the brief explicitly allows it. If cropping costs several accuracy
points, that means genuine clinical signal lives outside our mask — most likely the
**costophrenic angles** (the sharp corners where the diaphragm meets the ribcage, where
fluid pools) and the **retrocardiac region** (behind the heart). Segmentation models
routinely clip both. We would report that as a real finding about the segmentation,
and it goes straight into our limitations section.

### The honest-evaluation harness around it

This part is arguably as valuable as the change itself, because it is what the field skipped:

- **Patient-wise re-split** — all images from one child stay in one split. Otherwise the
  same patient appears in train *and* test and you are measuring memorisation. Gives us a
  real validation set instead of 16 images.
- **5 random seeds**, reporting **mean ± 95% CI** on every number.
- **Sensitivity/recall + AUC**, never accuracy alone (this is also a course rule).
- **ECE + a reliability diagram** — would be the first calibration numbers published on
  this benchmark.
- **Grad-CAM hit rate scored inside the lung mask** — measured, not displayed.

---


## 10 · YOUR THREE PAPERS — in depth

You have **the three ways people pushed past 96%**: combine models, explain models, or
throw out convolution entirely. Your slide is where the numbers peak — you present the
best results in the whole review, right before Harshal takes them apart.

**A framing that works well:** three genuinely different strategies, all landing in the
same 96–98% band, all on the same 624 images. The ceiling isn't moving, whatever you do.

---

### P7 · Chouhan et al., 2020, *Applied Sciences*
**"A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images"**
~890 citations

**The idea they are proposing.** This paper follows directly from something on Meghavi's
slide: Ayan & Ünver showed that different pretrained networks make **different mistakes**
— one is better at catching sick patients, another at clearing healthy ones. Chouhan's
argument: if their errors are different, **combine them and the errors partly cancel.**

**What they did.** Took **five** ImageNet-pretrained networks — **AlexNet, DenseNet-121,
Inception-V3, ResNet-18 and GoogLeNet** — used each as a feature extractor on the Kermany
X-rays, and combined their outputs into a single **ensemble** prediction.

*Ensemble in one sentence:* instead of trusting one doctor, you ask five and pool their
opinions. It only helps if they are wrong about *different* cases — which is exactly what
Ayan demonstrated.

**Results:** **96.4% accuracy** with **99.62% recall (sensitivity)**.

**Why recall is the number to quote — and this is a strong point to make.**
For a screening or triage tool, the two errors are not equally bad. A **false negative** —
telling a family a genuinely sick child is fine — can be fatal. A **false positive** just
means a doctor takes a second look. So sensitivity/recall is the metric that matters, and
99.62% means it almost never misses a case.

**⚠️ But be ready for the counterpoint, because it is a fair one.** Remember the dataset
is **2.7 : 1 in favour of pneumonia**. A model that leans toward answering "pneumonia"
will naturally score very high recall — a model that says "pneumonia" every single time
gets **100% recall** and is useless. So recall must always be read alongside specificity.
If she pushes on this, saying so yourself is much stronger than being caught by it.

**The honest cost.** Five networks means roughly **five times the compute and memory**,
at inference as well as training, for about **one point** over a single good DenseNet.
For a tool meant to run in an under-resourced clinic, that trade is not obviously worth it.

---

### P8 · Rajaraman et al., 2018, *Applied Sciences*
**"Visualization and Interpretation of Convolutional Neural Network Predictions in
Detecting Pneumonia in Pediatric Chest Radiographs"** · ~400 citations

**The idea they are proposing.** Everyone else is reporting accuracy. This paper argues
that in medicine that is not enough — you also have to show **what the network is
responding to**, or no clinician will ever trust it.

**What they did.** Built a "**customised**" VGG16 — they trimmed the standard architecture
and retrained it for this specific task rather than using it off the shelf — and paired it
with **CAM/saliency visualisations** of its predictions.

*CAM (Class Activation Mapping), in plain terms:* take the feature maps from the last
convolutional layer, weight each one by how strongly it pushed the final "pneumonia"
score up, add them together, and overlay the result on the X-ray as a heatmap.
Red = these pixels drove the answer.

**Results:** **96.2% accuracy** on detecting disease, **93.6%** on distinguishing
bacterial from viral.

**Why this is the most important paper on your slide — for our project specifically.**
This is the closest anyone using the Kermany dataset comes to asking *"is the model
actually looking at the lungs?"*

And it is **still only pictures.** They display heatmaps in a figure and invite you to
agree they look reasonable. **They never score them** — never measure the overlap against
a radiologist's own marking of where the disease is, never report a number.

That unmeasured gap is not a small omission. It is the exact hole our own experiment is
designed to fill, and Harshal's third paper proves that measuring it **changes the
answer**. When you present this one, land that point — you are setting up the punchline.

---

### P9 · Singh et al., 2024, *Scientific Reports*
**"Efficient Pneumonia Detection Using Vision Transformers on Chest X-rays"**

**The idea they are proposing.** Stop using convolution altogether. Replace it with
**self-attention** — the mechanism behind large language models.

**Vision Transformer (ViT), explained properly.** A CNN builds understanding *locally*:
each layer only sees a small neighbourhood, and it takes many stacked layers before any
part of the network can relate the top-left of an image to the bottom-right.

A ViT does something quite different:

1. Chop the image into a grid of small **patches** (typically 16×16 pixels).
2. Treat each patch like a **word in a sentence** — flatten it into a vector, and add a
   marker recording where in the image it came from.
3. Feed that sequence into a **Transformer**, whose **self-attention** mechanism lets
   **every patch directly compare itself with every other patch, in a single step.**

**Why that could genuinely help here.** Reading a chest X-ray often means *comparing*
regions — one lung field against the other, or a suspicious area against the surrounding
tissue. Those are long-range comparisons, and self-attention does them natively, whereas
a CNN has to build up to them gradually.

**Results:** **97.61% accuracy**, roughly **95% sensitivity** and **98% specificity** —
the highest reported on this benchmark.

**⚠️ The catch you must be ready for — she may well ask this.**
Vision Transformers are famously **data-hungry**; they typically need far more data than
CNNs because they have less built-in assumption about how images work (a CNN "knows"
that nearby pixels are related; a ViT has to learn even that). **5,216 training images is
tiny.** So how does it work at all?

**Answer: it is heavily pretrained** — on ImageNet-21k or similar — and only fine-tuned
here. Trained from scratch on 5,216 images, a ViT would lose badly to a small CNN.

**The other trade-off:** ViTs are computationally expensive. Published comparisons put
ViT inference **roughly two orders of magnitude slower than a small CNN on CPU** — which
matters a lot if the target is a clinic without a GPU.

---

### How to hand off to Harshal

End with something like:

> *"So three completely different strategies — ensemble five networks, visualise what one
> network sees, or throw out convolution entirely — and they all land between 96 and 98%
> on the same 624 images. The numbers look excellent. Harshal is going to explain why we
> don't believe them."*


---

## 11 · Your ~2 minutes — a script you can just say

*Not to be memorised word for word. Say it in your own words; this is the shape.*

> *"I'm covering three different ways people pushed past 96% — and they're genuinely
> different strategies, which is what makes it interesting.*
>
> ***Chouhan 2020*** *takes the finding from Meghavi's slide — that different networks make
> different mistakes — and does the obvious thing: ensembles five of them, AlexNet,
> DenseNet-121, Inception-V3, ResNet-18 and GoogLeNet. 96.4% accuracy with 99.62% recall.
> Recall is the number I'd focus on, because for a screening tool missing a sick child is
> far worse than a false alarm. Though I'd flag that our dataset is 2.7 to 1 toward
> pneumonia, so a model leaning that way gets high recall almost for free — you have to
> read it alongside specificity. And it's five times the compute for about one point.*
>
> ***Rajaraman 2018*** *is the one that matters most for our project. They argue accuracy
> alone isn't enough in medicine — you have to show what the network is responding to. So
> they build a customised VGG16, get 96.2%, and produce CAM heatmaps showing which pixels
> drove each prediction. This is the closest anyone using our dataset gets to asking
> whether the model is looking at the lungs. And it's still only pictures — they show the
> heatmaps, they never score them against a radiologist's marking. Hold that thought.*
>
> ***Singh 2024*** *throws out convolution entirely — a Vision Transformer. It chops the
> image into 16-by-16 patches, treats each like a word in a sentence, and uses
> self-attention so every patch can compare itself with every other patch in one step.
> A CNN needs many layers to relate opposite corners of an image; a transformer does it
> immediately. 97.61% accuracy — the highest on this benchmark.*
>
> *So: ensemble, explain, or replace the architecture. All three land between 96 and 98%,
> all on the same 624 images. Harshal will explain why we don't believe those numbers."*

---

## 12 · If you get asked something

**"Why would a Vision Transformer beat a CNN on only 5,000 images? ViTs need huge datasets."**
That's exactly right, and it's the key caveat. It only works because it's **heavily
pretrained** — ImageNet-21k or similar — and merely fine-tuned here. A ViT trained from
scratch on 5,216 images would lose badly to a small CNN, because a CNN has built-in
assumptions about images (nearby pixels are related) that a ViT has to learn from data.

**"How does self-attention actually work?"**
Every patch produces a query, a key and a value. Each patch's query is compared against
every other patch's key to get a set of weights — how relevant is each other patch to
me — and the patch's new representation is the weighted sum of all the values. The point
is that it's **all-to-all in one step**, so distant regions can be compared immediately.

**"Doesn't an ensemble of five models just overfit more?"**
Ensembling generally *reduces* variance rather than increasing it, so it usually helps
with overfitting. The real objection is cost — five times the compute and memory for
about one accuracy point, which is a bad trade for a tool meant to run in a
resource-limited clinic.

**"Isn't Grad-CAM already an explanation? Why do you need more?"**
It's a hypothesis, not evidence. Saporta 2022 — Harshal's third paper — scored seven
saliency methods against radiologists' own segmentations. Grad-CAM was the best of the
seven and **still significantly worse than a human**. So a convincing-looking heatmap
doesn't establish that the model used the right region; it has to be measured.

**"Which of your three would you build on?"**
Rajaraman, conceptually — it's the only one asking what the model is looking at. But we
picked Kermany as our base paper because it's the most cleanly specified and easiest to
reproduce faithfully.

**The universal escape hatch.** If you genuinely do not know:

> *"We didn't measure that — that's actually one of the things our experiment is set up
> to answer."*

This is a **strong** answer, not a weak one. Our entire presentation is an argument that
this field claims things it has not measured. Refusing to guess is us being consistent.
Never invent a number.

---

## 13 · Cheat card — the numbers worth memorising

| Number | What it is |
|--------|-----------|
| **5,856** | total images (5,216 train / 624 test / **16 val**) |
| **16** | the validation set. The scandal. |
| **624** | test images → **±1.5 pt** confidence interval |
| **2.7 : 1** | class imbalance (4,273 pneumonia / 1,583 normal) |
| **62.5%** | score you get by always answering "pneumonia" — our real floor |
| **1–5 years** | patient ages. One hospital, Guangzhou, China. |
| **92.8 / 93.2 / 90.1** | Kermany's accuracy / sensitivity / specificity — our reproduction target |
| **98.0%** | best reported on this benchmark (Rahman 2020) |
| **0.931 → 0.815** | Zech's AUC, internal → external hospital. The collapse. |
| **99.95%** | accuracy with which a CNN could identify *which hospital* an X-ray came from |
| **99.62%** | Chouhan's recall — but read it against the 2.7:1 imbalance |
| **97.61%** | Singh's ViT — the highest on this benchmark, heavily pretrained |

---

## 14 · Running order tomorrow

| Slide | Who | What |
|-------|-----|------|
| 1 | anyone | Title — topic, team |
| 2 | anyone | The dataset: 5,216 / 624 / **16**, one hospital, ages 1–5 |
| 3 | **Samarth** | Foundations — Kermany, CheXNet, ChestX-ray8 |
| 4 | **Meghavi** | The accuracy race — Stephen, Ayan & Ünver, Rahman |
| 5 | **Rachit** | Ensembles, interpretability, transformers — Chouhan, Rajaraman, Singh |
| 6 | **Harshal** | The reality check — Zech, DeGrave, Saporta |
| 7 | anyone | **The key chart** — accuracy climbed, evidence didn't |
| 8 | anyone | **The research gap** — the bonus-marks slide |
| 9 | anyone | What we build next — base paper, our one change, our prediction |

Total 10 minutes. Roughly 2 minutes each for slides 3–6, and the last three slides are
short. If someone does not want to speak, any of the others can take slide 7, 8 or 9 —
they are written to be presentable by anyone.

**One last thing.** If she pushes hard on any single point, the safest ground in the
entire presentation is this sentence:

> *"Every paper here reports one number on one small single-hospital test set, and nobody
> checked whether the model is looking at the lungs. That's our gap, and that's what we're
> going to measure."*

Good luck. You know this.
