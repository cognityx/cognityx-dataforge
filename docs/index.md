# Cognityx DataForge

DataForge turns evidence extracted from source documents into examples that
can be used to train or evaluate a model. It sits between document processing
and model training in the Cognityx application flow:

```text
+------------------------------+     +--------------------------------------+     +------------------------------+
|       COGNITYX INGEST        |     |         COGNITYX DATAFORGE           |     |      COGNITYX TRAINING       |
|                              |     |                                      |     |                              |
| Source Asset                 | --> | Evidence Selection                   | --> | Published Dataset            |
| Parsed Content               |     | Knowledge Unit                       |     | Training Experiment          |
| Chunk                        |     | Instruction / Question                |     | Evaluation                   |
| Evidence                     |     | Reference Answer                      |     | Adapter / Model Artifact     |
|                              |     | Training Example and Dataset Variant  |     |                              |
+------------------------------+     +--------------------------------------+     +------------------------------+

Provenance is recorded across the complete lifecycle.
```
evidence is physically prepared by Ingest but becomes meaningful for a particular training example when DataForge links it to that example.DataForge  selects one or more of evidence records and establishes:

```text
Evidence → supports → Knowledge unit
Evidence → supports → Reference answer
```



## Core Terms in Training Data Creation

The lifecycle of training-data creation starts from a source document or other source asset and ends with training examples, usually in the form of question–answer pairs or instruction–answer pairs for instruction fine-tuning.

Before describing this lifecycle, Cognityx defines the commonly used terms.

**Source asset** is the original digital object received or referenced by Cognityx, such as a PDF, Word document, web page, image, audio file, database record, or file-share URI.

**Source document** is a source asset whose content is treated as a document and can be parsed into text, sections, tables, pages, or other document structures.

**Evidence** is a specific part of the source content that supports a claim or answer. It includes the supporting content and enough location information to find it again.

**Provenance** records where the evidence came from and how it was created or processed. It includes its **lineage**, meaning the ordered chain connecting the evidence to the objects from which it was derived.

**Ground truth** is a fact, label, or expected result that has been validated and accepted as correct for a defined context.

**Reference answer** is an approved way of expressing the ground truth as an answer to a particular question or instruction.

### Illustrated example

A company travel-policy PDF is the **source asset** and, after being interpreted as a document, the **source document**.

The sentence stating that claims must be submitted within 30 days is the **evidence**.

The chain:

`PDF → parsed text → paragraph → evidence`

is the evidence **lineage**. The source version, processing steps, and this lineage together form its **provenance**.

The validated rule, “the submission deadline is 30 days,” is the **ground truth**.

The answer, “Employees must submit travel claims within 30 days of completing the journey,” is the **reference answer**.

The resulting question and reference answer can then become a training or evaluation example.

**Knowledge Unit** 
## Knowledge Unit

Instruction fine-tuning datasets are often created directly from source documents. The documents are divided into pages, paragraphs, or other smaller segments, and questions or instructions are generated from each segment.

In Cognityx, the basic addressable source segment is called an **evidence record**. Evidence is prepared and managed by Cognityx Ingest.

The simplest DataForge approach, called the **baseline variant**, generates a question or instruction directly from an evidence record.

However, a segment may contain several facts, rules, concepts, procedures, or relationships. Generating a question directly from the complete segment may therefore produce unclear, incomplete, or unfocused training examples.

DataForge introduces an intermediate step that first identifies what knowledge should be learned from the evidence. This extracted item is called a **knowledge unit**.

A knowledge unit is a self-contained fact, rule, concept, procedure, or relationship derived from one or more evidence records and suitable for generating training or evaluation examples.

A knowledge unit is not yet a question, instruction, answer, or training example. It represents the specific knowledge that DataForge intends to teach or test.

```text
Evidence
   |
   +--> Baseline variant: Question or instruction
   |
   +--> Knowledge unit --> Question or instruction
```


DataForge provides three ways to build examples: `paragraph-qa`,
`knowledge-unit-qa`, and the research recipe `knowledge-unit-probed-qa`. The
research recipe checks what an untrained base model already knows, compares its
responses with the source evidence, and creates validated question-and-answer
examples only where they are useful. Running a recipe again with the same
inputs and settings produces the same inspectable JSONL artifacts. This
repeatable behavior is technically called deterministic output.

## Variant 1: Paragraph QA Baseline

### Concept

The **Paragraph QA Baseline** follows the conventional method of creating instruction fine-tuning data directly from document content.

Each evidence record is divided into paragraph-sized spans. DataForge sends each paragraph to an inference model and asks it to generate one instruction and its corresponding answer.

```text
Evidence → Paragraph → Instruction and answer
```

This variant does not first discover a knowledge unit. The paragraph itself is treated as the material from which the training example is generated.

### Why Cognityx Retains This Variant

The Paragraph QA Baseline is deliberately simple. Its purpose is to provide a **control dataset** against which the more advanced Cognityx variants can be compared.

For example, Cognityx can later measure whether introducing knowledge units or model probing improves:

* instruction clarity;
* answer grounding;
* knowledge coverage;
* training usefulness; and
* model performance after fine-tuning.

The innovation is therefore not the direct paragraph-to-question method itself. Cognityx makes the baseline **traceable and experimentally reusable** by retaining the source evidence identifier, exact character range, source document, source asset, model configuration, prompt version, dataset identity, rejected generations, and resulting training record.

### Method

For each evidence record, DataForge:

1. identifies its paragraph spans;
2. sends each paragraph independently to the configured generation model;
3. requests one specific instruction and one answer grounded in that paragraph;
4. rejects empty, incomplete, or malformed model output;
5. converts successful output into the training message format;
6. retains the evidence ID and paragraph offsets in the record metadata; and
7. assigns the record to a training or evaluation split.

The generation prompt requires strict JSON containing `instruction` and `answer`. It also tells the model to return an empty result when the paragraph is too short or incomplete.

### Command

```bash
cognityx-dataforge build \
  --input-manifest storage://local-main/artifacts/ingest/runs/<run-id>/manifest.json \
  --dataset-name travel-policy-baseline \
  --recipe paragraph-qa \
  --config dataforge.toml \
  --storage-root /tmp/cognityx-storage
```

### Example Source Evidence

```text
Employees must submit travel claims within 30 calendar days after completing the journey.
```

### Example Generated Candidate

```json
{
  "instruction": "Within how many days must an employee submit a travel claim?",
  "answer": "An employee must submit the travel claim within 30 calendar days after completing the journey."
}
```

### Example Training Record

```json
{
  "record_id": "rec-...",
  "messages": [
    {
      "role": "user",
      "content": "Within how many days must an employee submit a travel claim?"
    },
    {
      "role": "assistant",
      "content": "An employee must submit the travel claim within 30 calendar days after completing the journey."
    }
  ],
  "split": "train",
  "metadata": {
    "recipe": "paragraph-qa",
    "source_asset_ids": ["asset-..."],
    "document_ids": ["doc-..."],
    "evidence_ids": ["evidence-..."],
    "char_start": 0,
    "char_end": 92,
    "generator_model": "<configured-model>",
    "prompt_versions": {
      "generation": "1.0"
    }
  }
}
```

### Example Command Result

```json
{
  "run_id": "job-...",
  "job_id": "job-...",
  "dataset_id": "dataset-...",
  "recipe": "paragraph-qa",
  "record_count": 1,
  "dataset_manifest_uri": "storage://local-main/datasets/<dataset-id>/<dataset-version>/manifest.json"
}
```


### Limitation

A paragraph may contain several facts, rules, concepts, or procedures. It may also contain background text that is not itself useful knowledge.

Because this variant generates an instruction directly from the paragraph, the resulting example may be broad, unfocused, incomplete, or may test only an incidental part of the paragraph.

This limitation motivates the next variant: **Knowledge-Unit QA**.



- Storage and Jobs as the two operational foundations that make large, repeatable experiments manageable.
- Inference as the model-execution foundation that makes the recipes hardware-aware and provider-neutral.
- Ingest as the upstream evidence producer.
- Training as the downstream dataset consumer.

```test
+----------------+      +-----------------------------------+      +----------------+
| COGNITYX       |      | COGNITYX DATAFORGE                |      | COGNITYX       |
| INGEST         | ---> | Dataset-building worker           | ---> | TRAINING       |
| Evidence       |      | Recipes, validation, finalization |      | Experiments    |
+----------------+      +-----------------------------------+      +----------------+
                                  |          |          |
                                  v          v          v
                           +-----------+ +---------+ +-----------+
                           | STORAGE   | | JOBS    | | INFERENCE |
                           | Artifacts | |Lifecycle| | Model calls|
                           +-----------+ +---------+ +-----------+
```



The V0 boundary is deliberately small: DataForge owns dataset construction,
validation, generation, rejection records, and dataset manifests. Ingest owns
document parsing, Storage owns durable artifacts, Jobs owns lifecycle state,
and Inference owns model serving.

Read the [reference guide](reference.md) for the end-to-end command-line sequence.
