AI4MARS Mars Rover Semantic Segmentation

Project Handoff Summary — Dataset Exploration and Preprocessing

1. Project Overview

We are working on a Computer Vision project using the AI4MARS dataset for Mars rover terrain semantic segmentation.

The objective is to develop a deep learning model that can perform pixel-level terrain classification from Mars rover images.

This is a semantic segmentation task, not ordinary image classification.

Each input image has a corresponding pixel-level label mask. Every labeled pixel belongs to one of four terrain categories, or is marked as unlabeled.

The dataset is hosted on Hugging Face:

hassanjbara/AI4MARS

We load it directly using:

from datasets import load_dataset
dataset = load_dataset("hassanjbara/AI4MARS")

Important: We are using the Hugging Face dataset directly. Do not assume that the dataset has been downloaded into a manually organized local data/ directory.

⸻

2. Project Structure

The current project structure is:

mars-rover-segmentation/
│
├── data/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_experiments.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│
├── README.md
└── requirements.txt

The first two notebooks are the current focus:

* 01_dataset_exploration.ipynb: Dataset understanding and exploratory analysis.
* 02_data_preprocessing.ipynb: Dataset integrity validation and preprocessing.

The project has not yet moved into model development.

⸻

PART A — DATASET EXPLORATION

3. Dataset Structure

The dataset contains four splits:

Split	Number of Samples
train	18,130
test_min1	322
test_min2	322
test_min3	322
Total	19,096

Each split contains the following features:

Feature	Description
image	Original Mars rover image
label_mask	Pixel-level terrain annotations
rover_mask	Mask identifying the rover’s own body
range_mask	Mask identifying regions beyond 30 meters
has_masks	Whether auxiliary masks are available
has_labels	Whether a terrain label mask is available

All images and masks are represented as Hugging Face Image features.

4. Training Annotation Distribution

The training split contains:

has_masks	has_labels	Samples
False	False	33
False	True	1
True	False	2,033
True	True	16,063

Therefore:

* Total training samples: 18,130
* Labeled samples: 16,064
* Unlabeled samples: 2,066

Approximately 88.60% of the training samples have both masks and labels.

The three test splits each contain 322 samples, and all samples in those splits have both masks and labels.

5. Image and Mask Properties

An inspection of 200 samples from each split showed:

Property	Result
Image dimensions	1024 × 1024
Image mode	L
Mask dimensions	1024 × 1024
Mask mode	L

All inspected images and masks were grayscale.

The dimensions of the original images and masks are aligned.

6. Terrain Class Semantics

The dataset uses the following pixel values:

Pixel Value	Terrain Class
0	Soil
1	Bedrock
2	Sand
3	Big Rock
255	NULL / Unlabeled

There are four actual terrain classes, not five.

The value 255 represents unlabeled pixels and must be treated as an ignore value.

The dataset documentation defines Big Rock as a rock greater than 30 cm tall and at least 50 cm wide.

The unique pixel values found in the inspected label masks were:

[0, 1, 2, 3, 255]

No unexpected label values were found.

The auxiliary masks use binary values:

* 0: Background / not excluded
* 1: Region to be excluded

7. Initial Pixel Distribution

An initial inspection of 100 labeled training masks produced:

Pixel Value	Number of Pixels
0	21,128,115
1	30,323,671
2	6,671,689
3	917,845
255	45,816,280

A later full pixel audit across all dataset splits produced the following distribution:

Pixel Value	Class	Total Pixels	Masks Containing Value
0	Soil	3,714,545,901	8,507
1	Bedrock	4,791,786,478	10,886
2	Sand	1,239,993,686	5,813
3	Big Rock	86,815,188	2,304
255	NULL	8,024,108,027	17,030

No unexpected pixel values were discovered.

The distribution shows substantial class imbalance. Big Rock is particularly underrepresented compared with the other terrain classes.

The full distribution above is the original label-mask distribution, before applying rover and range exclusions.

8. Annotation Consistency

We also checked whether the has_labels metadata agreed with the actual availability of label_mask.

Result:

Annotation inconsistencies: 0

All samples were consistent with their metadata.

This confirms that the label availability metadata can be used reliably when identifying labeled training samples.

⸻

PART B — DATASET INTEGRITY AUDIT

9. Why We Conducted an Integrity Audit

Before preprocessing, we needed to establish whether:

* Original images were corrupted.
* Label masks were missing.
* Rover masks were missing.
* Range masks were missing.
* Images and masks had matching dimensions.
* Any unexpected mask values existed.

The integrity audit was implemented in 02_data_preprocessing.ipynb.

We disabled automatic Hugging Face image decoding using:

from datasets import Image as HFImage
raw_datasets = {}
image_columns = [
    "image",
    "label_mask",
    "rover_mask",
    "range_mask"
]
for split, data in dataset.items():
    raw_data = data
    for column in image_columns:
        raw_data = raw_data.cast_column(
            column,
            HFImage(decode=False)
        )
    raw_datasets[split] = raw_data

This allowed us to inspect image bytes and paths directly and identify corrupted files without relying on automatic decoding.

10. Missing and Corrupted Data

The integrity audit produced:

Split	Feature	Missing	Corrupted
train	image	0	3
train	label_mask	2,066	0
train	rover_mask	33	0
train	range_mask	34	0
test_min1	image	0	0
test_min1	label_mask	0	0
test_min1	rover_mask	0	0
test_min1	range_mask	0	0
test_min2	image	0	0
test_min2	label_mask	0	0
test_min2	rover_mask	0	0
test_min2	range_mask	0	0
test_min3	image	0	0
test_min3	label_mask	0	0
test_min3	rover_mask	0	0
test_min3	range_mask	0	0

Three corrupted original images were identified in the training split:

[8990, 12577, 17053]

No corrupted images or masks were found in the three test splits.

Important observations

* There are 2,066 training samples without label masks.
* There are 33 training samples without rover masks.
* There are 34 training samples without range masks.
* Three training images are corrupted.

The missing auxiliary masks are separate from the missing label masks. Their overlap must be considered when designing preprocessing.

11. Image and Mask Dimension Verification

We inspected the dimensions of the images and masks.

The initial inspection showed that all 200 sampled images from each split were 1024 × 1024 grayscale images.

We also performed a full training image-mask alignment audit.

Result:

Split	Matching	Mismatched	Missing Masks	Corrupted
train	16,064	0	2,063	3
test_min1	322	0	0	0
test_min2	322	0	0	0
test_min3	322	0	0	0

All valid image-label pairs had matching dimensions.

There were no image-mask dimension mismatches.

The 2,063 missing masks in this alignment table are valid training images without label masks. The remaining three corrupted image rows could not be treated as valid image-mask pairs.

The earlier missing label_mask count of 2,066 therefore reconciles as:

2,063 missing label masks
+ 3 corrupted image rows
= 2,066 unusable image-label rows

⸻

PART C — MASK PREPROCESSING REQUIREMENTS

12. Official Preprocessing Instructions

The project guide PDF, Space_ML_Project_Guides.pdf, provides explicit instructions for AI4MARS preprocessing.

Each dataset row contains:

* The original rover image.
* The terrain label mask.
* The rover mask, identifying the rover’s own body.
* The range mask, identifying regions beyond 30 meters.

The guide explicitly instructs us to:

Set the rover’s body and everything beyond 30 meters to the no-label value.

This is an important preprocessing requirement.

We must not simply train on the original label masks without applying these exclusions.

13. Required Mask Processing

The preprocessing pipeline must handle four things:

1. The four actual terrain classes.
2. Existing unlabeled pixels (255).
3. Rover body regions.
4. Regions beyond 30 meters.

The intended behavior is:

A. Preserve the four terrain classes

Keep the original class IDs:

0 = Soil
1 = Bedrock
2 = Sand
3 = Big Rock

Do not remap them unnecessarily.

B. Preserve existing unlabeled pixels

Pixels already labeled 255 must remain 255.

C. Exclude rover body regions

For the rover mask:

label[rover_mask > 0] = 255

D. Exclude regions beyond 30 meters

For the range mask:

label[range_mask > 0] = 255

Both exclusions must be applied to the terrain label mask.

The original image should remain unchanged. We are modifying the target label mask, not deleting image pixels.

Important implementation requirements

* Convert masks into NumPy arrays.
* Make a copy of the original label mask before modifying it.
* Ensure the label mask and auxiliary masks have matching dimensions.
* Use mask > 0 to identify excluded regions, following the project guide.
* If an auxiliary mask is None, handle it safely rather than attempting to decode it.
* Preserve the original class IDs and the ignore value.
* Apply rover and range exclusions before resizing.
* When resizing segmentation masks later, use nearest-neighbor interpolation. Never use bilinear or bicubic interpolation for categorical label masks.

The project guide’s example implementation follows this logic:

def prepare(row):
    image = np.array(row["image"].convert("RGB"))
    label = np.array(row["label_mask"])
    label = label[..., 0] if label.ndim == 3 else label
    label = label.astype(np.int64)
    for key in ("rover_mask", "range_mask"):
        if row[key] is not None:
            m = np.array(row[key])
            m = m[..., 0] if m.ndim == 3 else m
            label[m > 0] = 255
    return image, label

This is the reference behavior from the project guide, not necessarily the final production implementation. Add appropriate validation and error handling.

14. Handling Missing Auxiliary Masks

There are some training samples with missing auxiliary masks:

* 33 missing rover masks.
* 34 missing range masks.

The project guide’s example skips an exclusion when its corresponding mask is None.

This is a possible approach, but the preprocessing implementation should explicitly track these cases.

Do not silently claim that all rover and range regions were excluded when the corresponding mask was unavailable.

A decision is needed about how to handle these samples in the final training pipeline. At minimum, record the missing-mask counts and make the behavior transparent.

Do not automatically discard all such samples without evaluating the consequences.

15. Expected Preprocessing Validation

After applying the exclusions, validate the resulting masks.

The validation should establish:

Label integrity

* Only 0, 1, 2, 3, and 255 are present.
* No unexpected class IDs have been introduced.

Exclusion correctness

* Every pixel where rover_mask > 0 becomes 255.
* Every pixel where range_mask > 0 becomes 255.
* Pixels already labeled 255 remain 255.

Shape integrity

* The processed label mask retains the original dimensions before resizing.
* The original image and processed label mask remain spatially aligned.

Data integrity

* Original source masks are not modified.
* The original images are not modified.
* Missing auxiliary masks are tracked.

Distribution analysis

* Calculate the class distribution after exclusions.
* Compare the distribution before and after preprocessing.
* Track the number and percentage of ignored pixels.
* Observe how much each exclusion contributes to the ignored-pixel count.

A useful visualization should compare:

1. Original rover image.
2. Original terrain label mask.
3. Rover mask.
4. Range mask.
5. Final processed label mask.

Use a clear discrete color mapping so the four terrain classes and the ignore region are visually distinguishable.

⸻

PART D — CURRENT PROJECT STATUS

16. Completed Work

Notebook 01 — Dataset Exploration

Completed:

* Loaded the Hugging Face dataset.
* Inspected all dataset splits.
* Examined annotation availability.
* Inspected image and mask dimensions.
* Identified grayscale image format.
* Visualized sample images and masks.
* Identified terrain label IDs.
* Examined auxiliary mask values.
* Calculated initial pixel distributions.
* Checked annotation metadata consistency.
* Inspected test split annotation availability.

Notebook 02 — Dataset Integrity Audit

Completed:

* Disabled automatic image decoding.
* Checked missing and corrupted files.
* Identified three corrupted training images.
* Identified missing label and auxiliary masks.
* Verified image dimensions.
* Verified image-mask alignment.
* Confirmed there are no unexpected label values.

## Preprocessing: Completed

**Pipeline:** raw row → safe decode → rover/range exclusions on a copy of the label →
nearest-neighbor resize (labels) / bilinear (images) → RGB + ImageNet normalization → tensors.

**Findings**
- Rover (`rover_mask > 0`) and >30 m (`range_mask > 0`) regions are already 255 in the
  source labels: enforced on all 16,064 usable train rows and 966 test rows, 0 pixels changed.
- 3 corrupted images are all unlabeled. Only train row 7211 lacks a range mask (kept, flagged).
- 163 usable train rows are 100% ignored (no signal) and were dropped from train/val;
  test splits untouched.
- Class shares among labeled train pixels: Soil 37.2%, Bedrock 49.8%, Sand 12.0%,
  Big Rock 0.9% (severe imbalance).

**Splits** (`data/splits.json`): block-wise (100 consecutive rows) 90/10, seed 42.
Train 14,312 | Val 1,589 | test_min1/2/3: 322 each. Block-wise is a heuristic against
near-duplicate leakage (no sequence IDs exist in the dataset).

**Code:** `src/data/preprocessing.py` (`apply_exclusions`, `resize_label`, `Ai4MarsSegmentation`).
Image size 512x512. Labels are int64 with 255 as the ignore index.

**For modeling:** use `ignore_index=255` in the loss, and plan for Big Rock imbalance
(class weights, Dice/focal loss, or per-class IoU as the primary metric).

