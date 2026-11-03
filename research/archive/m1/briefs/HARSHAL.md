# Harshal Nerpagar — read this before tomorrow

**Roll no:** 230076 · **You present:** slide 6, *The reality check* · **Your time:** ~2 minutes

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
4. **You present the reality check** — the three papers that prove point 3, and you hand
   straight into the research gap. **Yours is the most important slide in the talk.**

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

You have **the reality check**, and it is the most important slide in the presentation.
Everything before you builds up numbers; **you take them apart**, and you hand straight
into the research gap that earns the bonus marks.

Two things make your slide land:

1. **These are the heaviest papers in the review** — *PLOS Medicine* and *Nature Machine
   Intelligence*, not workshop proceedings. Say the venues out loud.
2. **None of the three uses our dataset.** That is not a weakness in your slide — it *is*
   our research gap. The people doing the checking and the people using our benchmark are
   two separate communities who never talk to each other.

---

### P10 · Zech et al., 2018, *PLOS Medicine*
**"Variable Generalization Performance of a Deep Learning Model to Detect Pneumonia in
Chest Radiographs: A Cross-Sectional Study"**

**The idea they are proposing.** Everyone reports how well their model does on data from
the same hospital it was trained on. Zech asked the obvious next question that nobody was
asking: **what happens when you take it to a different hospital?**

**What they did.** Three sites:
- **MSH** — Mount Sinai Hospital, 42,396 radiographs
- **NIH** — the National Institutes of Health Clinical Center, 112,120 radiographs
- **IU** — Indiana University Network for Patient Care, 3,807 radiographs

They trained on one or two sites and tested on the held-out one.

**Results (AUC — remember, 0.5 is a coin flip, 1.0 is perfect):**

| Trained on | Internal | External | What happened |
|---|---|---|---|
| MSH | **0.802** | 0.717 at NIH | significantly worse |
| NIH | **0.750** | 0.695 at MSH | significantly worse |
| MSH + NIH jointly | **0.931** | **0.815** at Indiana | significantly worse |

Note the last row especially: training on two sites gave the **best** internal score of
all — 0.931 — and it still fell to 0.815 on a genuinely new hospital. **Pooling more data
did not fix generalisation.**

**The smoking gun — this is the detail that makes the room go quiet.**
They trained a CNN on a different task: not "does this patient have pneumonia" but
**"which hospital did this X-ray come from."** It hit **99.95% accuracy for NIH and
99.98% for MSH.** It could also separate **portable from fixed X-ray machines with 100%
accuracy.**

**Why the model can do that.** Metal laterality tokens (the little "L"/"R" markers) placed
differently at each site; different scanner brands — inpatient wards used Konica Minolta
while emergency departments used Fujifilm, which even inverts the grayscale; burnt-in text
labels; differences in patient positioning.

**And here is the trap — make sure you explain this bit, it is the heart of it.**
Portable X-ray machines get wheeled to patients who are **too sick to walk to radiology**.
So "portable" correlates strongly with "seriously ill." A model can therefore score well
by **detecting the machine** rather than the disease. That is a *real* statistical pattern
in the data — it is not a bug in their code — and it is completely worthless in a new
hospital where the equipment habits are different.

**Your one-line takeaway:** *it learned the hospital, not the disease.*

---

### P11 · DeGrave, Janizek & Lee, 2021, *Nature Machine Intelligence*
**"AI for Radiographic COVID-19 Detection Selects Shortcuts Over Signal"**
~980 citations

**The idea they are proposing.** Zech showed *that* generalisation fails. This paper
explains **why**, and gives the failure its name: **shortcut learning**.

**Shortcut learning, in plain terms.** A neural network is lazy in a very specific way.
It is not trying to understand pneumonia — it is minimising its error, and it will latch
onto **whatever feature most easily predicts the label** in the data you gave it. If a
shortcut exists, it takes it, because a shortcut is **cheaper to learn** than actual
pathology. It is not cheating; it is doing exactly what we asked.

**What they did.** Took COVID-19 chest X-ray classifiers that reported excellent results,
and interrogated them using explanation techniques and generative models — essentially
asking the model to show what it would change in an image to flip its own decision.

**What they found.** The models were keying on **laterality markers, image borders,
collimation edges** (the black frame produced by the X-ray beam's aperture), **and patient
positioning** — features **entirely outside the lungs**. On external data, performance
collapsed.

**Why it happened so severely there.** Many COVID datasets were assembled by combining a
COVID-positive collection from one source with a negative collection from a different
source. So **the source of the image perfectly predicted the label**. The easiest possible
shortcut was handed to the model on a plate.

**How to connect this to our project — this is your best line, deliver it deliberately.**

Someone may reasonably object: *"but Kermany is a single hospital, so there's no
site confound — isn't it safer?"*

**It is arguably worse, not better.** With a single site, **nothing in the data ever
punishes the model for using acquisition artifacts.** There is no second hospital whose
different markers and machines would force it to ignore them. So shortcuts are free — the
model can use them at no cost to its training score. And **nobody has ever tested whether
it does**, because nobody externally validates on this dataset.

---

### P12 · Saporta et al., 2022, *Nature Machine Intelligence* — **CheXlocalize**
**"Benchmarking Saliency Methods for Chest X-Ray Interpretation"**

**The idea they are proposing.** Every paper in this field shows Grad-CAM heatmaps and
says "look, the model is attending to the right region." Saporta's response: **stop
showing them and start scoring them.**

**What they did.** Took **seven** saliency methods (Grad-CAM and six others), had
**radiologists segment the actual pathology** on chest X-rays to create ground truth, and
measured how well each method's heatmap matched, using two metrics:

- **mIoU (mean Intersection over Union)** — area of overlap ÷ area of union. **Strict**:
  you must get the whole extent of the finding right, not just touch it.
- **Hit rate** — much looser: did the hottest point of the heatmap land inside the true
  region at all?

They also established a **human benchmark**: how well one radiologist's segmentation
matches another's, which is the realistic ceiling.

**Results:**

1. **Grad-CAM localised better than the other six methods** — so the field's default
   choice was at least the right one.
2. **All seven performed significantly worse than the human benchmark.**
3. The gap was **largest for pathologies that were small and complex in shape** — which
   is exactly what early or subtle pneumonia looks like.
4. **Model confidence correlated with localisation quality** — when the model was less
   sure, its heatmap was also worse.

**Why this is the paper that closes our argument.** Papers P1 through P9 — every single
one on the first three slides — display Grad-CAM figures as evidence that their model
looks in the right place. Saporta demonstrates that when you actually **measure** that
claim, it frequently does not hold.

**Your line:** *a heatmap is a hypothesis, not evidence.*

---

### How to hand off to the gap slide

End with something like:

> *"So: models that collapse when you move hospitals, a named mechanism explaining why —
> they use whatever's easiest, and it's usually outside the lungs — and proof that the
> heatmaps everyone shows as evidence don't survive being measured. Now here's the
> problem. None of these three papers uses our dataset. And none of the nine papers that
> do use our dataset ran any of these checks. Those two literatures never touch — and
> that's our research gap."*


---

## 11 · Your ~2 minutes — a script you can just say

*Not to be memorised word for word. Say it in your own words; this is the shape.*

> *"The first three slides were about numbers going up. I'm covering the papers that
> asked whether any of it holds up — and these are the heavyweight venues, PLOS Medicine
> and Nature Machine Intelligence.*
>
> ***Zech 2018.*** *They did the thing nobody else was doing: took the model to a
> different hospital. Trained on Mount Sinai and NIH together, got AUC 0.931 in-house —
> the best internal score in the paper — and it dropped to 0.815 at Indiana. But the
> detail that really gets you is this: they trained a network to predict which hospital an
> X-ray came from, and it was 99.95% accurate. It could tell portable from fixed X-ray
> machines with 100% accuracy. And portable machines get wheeled to patients too sick to
> walk to radiology — so the model can score well by spotting the machine instead of the
> disease. It learned the hospital, not the disease.*
>
> ***DeGrave 2021*** *names the mechanism: shortcut learning. A network isn't trying to
> understand pneumonia, it's minimising error — so it grabs whatever feature most easily
> predicts the label. They showed COVID classifiers keying on laterality markers, image
> borders, collimation edges, patient positioning. All outside the lungs. And you might
> think a single-hospital dataset like ours is safer — I'd argue it's worse, because
> there's no second site to punish the model for using those artifacts. Shortcuts are
> free, and nobody's tested whether ours uses them.*
>
> ***Saporta 2022*** *closes it. Everyone shows Grad-CAM heatmaps as proof the model
> looked in the right place. Saporta scored seven saliency methods against radiologists'
> own segmentations. Grad-CAM was the best of the seven — and still significantly worse
> than a human, worst of all on small, complex findings. So a heatmap is a hypothesis,
> not evidence.*
>
> *Now here's the problem. None of these three papers uses our dataset. And none of the
> nine papers that do use our dataset ran any of these checks. Those two literatures never
> touch — and that's our research gap."*

---

## 12 · If you get asked something

**"But Kermany is a single hospital, so there's no site confound — isn't that safer?"**
It's arguably **worse**, not better. With a single site, nothing in the data ever punishes
the model for using acquisition artifacts — there's no second hospital whose different
markers and machines would force it to ignore them. So shortcuts are free. And nobody has
tested whether it happens, because nobody externally validates on this dataset.

**"What exactly is AUC and why do these papers use it instead of accuracy?"**
The model outputs a score from 0 to 1. Accuracy makes you pick a cutoff; AUC measures
quality across every possible cutoff. Concretely: pick one random sick patient and one
random healthy patient — AUC is the probability the model scores the sick one higher.
0.5 is a coin flip, 1.0 is perfect. It's preferred here because it doesn't depend on an
arbitrary threshold and handles class imbalance better.

**"Is a drop from 0.931 to 0.815 really that bad?"**
Yes, in this context. That's the model at its best internally falling to something much
closer to marginal on genuinely new data — and it's the *joint* model, trained on two
hospitals, which was supposed to be the robust one. Pooling more data didn't fix it.

**"How do you know these models are using shortcuts and not the actual disease?"**
DeGrave showed it directly using explanation and generative techniques — essentially
asking the model what it would change in the image to flip its own decision, and finding
it pointed to markers and borders. Zech showed it indirectly: the same architecture could
identify the hospital with 99.95% accuracy, so that information is unmistakably in the
image and available to be used.

**"So is Grad-CAM useless?"**
No — it was the best of the seven methods Saporta tested, so it's the right tool. The
problem is treating an unmeasured heatmap as proof. It's a hypothesis about what the model
used; it needs to be scored against expert annotation before it counts as evidence.

**"Why does it matter that these three papers don't use your dataset?"**
That's precisely our gap. The people doing rigorous checking and the people using our
benchmark are two separate communities. Nobody has run these checks on the dataset that
eight of our twelve papers report results on.

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
| **0.802 → 0.717** | Zech, MSH-trained, internal → NIH |
| **100%** | accuracy separating portable from fixed X-ray machines |
| **7 methods** | Saporta's saliency benchmark — all worse than a radiologist |

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
