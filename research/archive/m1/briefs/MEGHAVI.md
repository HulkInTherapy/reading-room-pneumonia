# Meghavi Rao — read this before tomorrow

**Roll no:** 230044 · **You present:** slide 4, *The accuracy race* · **Your time:** ~2 minutes

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
4. **You present the accuracy race** — three papers that each swap in a different
   pretrained network and report a higher number, on the same 624 test images.

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

You have **the accuracy race**. Your three papers are the pattern of the entire field:
same 5,856 images, different pretrained network each time, a higher number reported each
time. Your job is to make that pattern visible — and to plant the first seeds of doubt.

**A framing that works well:** you are showing three *different strategies* for the same
task — build one from scratch, compare two off-the-shelf, or sweep four and pick the
winner — and all three land in the same place without anyone checking their work.

---

### P4 · Stephen et al., 2019, *Journal of Healthcare Engineering*
**"An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare"**
~800 citations

**The idea they are proposing.** Everyone else assumes you need transfer learning.
Stephen's paper challenges that: *what if you just build a CNN from scratch and lean hard
on data augmentation instead?*

**What they did.** Constructed a convolutional network from the ground up — stacked
convolution and pooling blocks feeding a dense classifier head — with **no ImageNet
weights at all**. To compensate for having only 5,216 training images, they used
aggressive **data augmentation**: rotation, zoom, shifting, flipping, rescaling.

*Why augmentation substitutes for pretraining:* a big network can simply memorise 5,216
images (**overfitting**). If instead every epoch shows it slightly rotated, zoomed,
shifted versions, it can never memorise exact pixels — it is forced to learn the actual
pattern. Effectively you have multiplied your dataset.

**Result.** Competitive accuracy on the Kermany data, with no pretrained weights.

**Why this paper is more interesting than it looks — this is your good line.**
If a from-scratch CNN can match a carefully pretrained one, then the "improvements" in
this field are probably **not** about clever architecture at all. It suggests the task,
on this dataset, is easy enough that many approaches saturate at the same place. That is
a hint of something we make explicit later in the talk.

**What is missing.** No confidence interval, no external test set, and the handling of
the tiny validation split is loosely described.

---

### P5 · Ayan & Ünver, 2019, IEEE EBBT
**"Diagnosis of Pneumonia from Chest X-Ray Images Using Deep Learning"**
~550 citations

**The idea they are proposing.** A straight, honest head-to-head. Same dataset, two
well-known pretrained networks, report both fairly and see which wins.

**What they did.** Fine-tuned **VGG16** and **Xception** on the Kermany data and compared
them on the same test set.

*The two networks, in one line each:*
- **VGG16** — an old, simple design: a plain deep stack of small 3×3 filters. Not clever,
  but very sturdy and well understood.
- **Xception** — uses *depthwise separable convolutions*: it splits each convolution into
  two cheaper steps, first looking across space, then across channels. More efficient per
  parameter.

**Results:**

| | Accuracy | Specificity | Sensitivity |
|---|---|---|---|
| **VGG16** | **87%** | **91%** | 82% |
| **Xception** | 82% | 76% | **85%** |

**The genuinely interesting finding — lead with this, not the accuracy.**
They do not just differ in overall score, they **differ in the kind of mistake they make**.
VGG16 is better at correctly clearing healthy children (higher specificity). Xception is
better at catching sick ones (higher sensitivity). **The two networks are wrong about
different patients.**

**Two reasons this matters:**

1. **It is the most believable result in your group.** Notice that 87% and 82% are *far*
   below the 96–98% everyone else reports on the identical dataset. That should make you
   suspicious of the higher numbers — and it is a great thing to say out loud, because it
   sets up the gap slide.
2. **"Different models, different errors" is precisely the justification for ensembles** —
   which is the first paper on Rachit's slide. That is your natural hand-off.

**What is missing.** They *observe* the sensitivity/specificity trade-off but never
explain **why** — there is no analysis of what each network is actually keying on. That
"why" is exactly the question our own experiment is designed to answer.

---

### P6 · Rahman et al., 2020, *Applied Sciences*
**"Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia
Detection Using Chest X-ray"** · ~815 citations

**The idea they are proposing.** Be systematic. Instead of arguing for one architecture,
take four well-known pretrained networks, run them all on the same data under the same
conditions, and report which actually wins.

**What they did.** Compared **AlexNet, ResNet-18, DenseNet-201 and SqueezeNet**, fine-tuned
on the Kermany X-rays. They ran two tasks: the binary normal-vs-pneumonia problem, and the
harder three-way normal / bacterial / viral problem.

*The architectures, briefly:* **AlexNet** is the 2012 original that started deep learning
for vision. **ResNet** introduced *skip connections* — a layer can pass its input forward
unchanged, which is what allowed networks to get genuinely deep without training
collapsing. **DenseNet** takes that idea to its limit by connecting every layer to every
later layer. **SqueezeNet** is deliberately tiny, designed for low-power devices.

**Results.** **DenseNet-201 wins**, at roughly **98% on normal vs pneumonia** — the
highest number reported anywhere on this benchmark, and the one most often quoted.

**But look at the second number, and make sure you say this.** On the harder task the
same model drops to about **95% on bacterial and 93.3% on viral** pneumonia.

**Why that second number is the clinically important one.** Bacterial pneumonia needs
antibiotics; viral pneumonia does not. Detecting *that something is wrong* is the easy
half of the problem. The half that actually changes a child's treatment is harder, and
**every paper in our review is worse at it.** That is a nice, quietly devastating point.

**What is missing.** Single split, single seed, no confidence interval, no external
validation. The 98% headline is exactly the kind of number our gap slide is about.

---

### How to hand off to Rachit

End with something like:

> *"So: three different strategies — build from scratch, compare two, sweep four — and
> they all land in the same 87-to-98 band, on the same 624 images, with nobody reporting
> error bars. And since Ayan showed different networks make different mistakes, the
> obvious next move is to combine them. That's where Rachit picks up."*


---

## 11 · Your ~2 minutes — a script you can just say

*Not to be memorised word for word. Say it in your own words; this is the shape.*

> *"I'm covering what happened once everyone had the dataset — I'd call it the accuracy
> race. Three papers, three strategies, same 624 test images.*
>
> ***Stephen 2019*** *asks whether you even need transfer learning. They build a CNN from
> scratch, no ImageNet weights at all, and compensate with heavy data augmentation —
> rotating, zooming and flipping the training images so the network can't just memorise
> them. It comes out competitive. I think that's more interesting than it sounds: if a
> from-scratch network matches a carefully pretrained one, the gains in this field
> probably aren't about clever architecture at all.*
>
> ***Ayan and Ünver 2019*** *is the most honest paper in my set. Straight head-to-head:
> VGG16 gets 87% accuracy, Xception 82%. But the useful result is the split — VGG16 is
> better at correctly clearing healthy children, 91% specificity, while Xception is better
> at catching sick ones, 85% sensitivity. They're wrong about different patients. And
> notice these numbers are well below the 96 to 98% everyone else reports on the exact
> same dataset — that's worth holding on to.*
>
> ***Rahman 2020*** *sweeps four backbones — AlexNet, ResNet-18, DenseNet-201,
> SqueezeNet. DenseNet-201 wins at about 98%, which is the highest number anyone reports
> on this benchmark. But look at their second result: on bacterial versus viral it drops
> to 93.3%. That's the question that actually decides whether a child gets antibiotics —
> and every paper we read is worse at it than at the easy question.*
>
> *So three strategies, all landing in the same band, on the same 624 images, and nobody
> reporting error bars. And since Ayan showed different networks make different mistakes,
> the obvious next step is to combine them — which is where Rachit starts."*

---

## 12 · If you get asked something

**"Isn't 98% basically solved? Why are you still working on this?"**
It's 98% on 624 images from one hospital, with no confidence interval, one seed, one
split, and no external validation. And the same paper drops to 93.3% on bacterial vs
viral, which is the question that actually changes treatment. High accuracy on an easy
single-centre benchmark isn't the same as a working tool.

**"Why do VGG16 and Xception differ in sensitivity and specificity?"**
The paper doesn't say — they report the trade-off but never analyse it. That missing
"why" is honestly part of our motivation: nobody in this literature investigates what the
networks are actually keying on.

**"What is data augmentation and why does it substitute for more data?"**
You make realistic modified copies of your training images — rotate, zoom, shift, flip.
The lungs are still lungs and the label is still correct, but the network can't memorise
exact pixels any more, so it's forced to learn the underlying pattern instead of the
specific images.

**"Why is bacterial vs viral harder than normal vs pneumonia?"**
Because both classes have visible disease — you're distinguishing between two abnormal
appearances rather than abnormal from normal. Classically bacterial tends to be a dense
focal patch and viral more diffuse and spread out, but they overlap heavily in practice,
especially in young children.

**"Why is Ayan's 87% so much lower than everyone else's 96–98% on the same data?"**
Honestly, we can't fully tell from the papers — it likely comes down to differences in
preprocessing, augmentation, and how the split was handled. Which is itself part of our
point: these numbers aren't strictly comparable, and nobody reports enough detail to know.

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
| **87% / 82%** | Ayan's VGG16 vs Xception — the honest, believable numbers |
| **93.3%** | Rahman's bacterial-vs-viral — the clinically important, harder task |

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
