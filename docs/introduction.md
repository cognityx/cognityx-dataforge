## Core Terms in Training Data Creation

The lifecycle of training-data creation starts from a source document or other source asset and ends with training examples, usually in the form of question–answer pairs or instruction–answer pairs for instruction fine-tuning.

Before describing this lifecycle, Cognityx defines the commonly used terms.

**Source asset** is the original digital object received or referenced by Cognityx, such as a PDF, Word document, web page, image, audio file, database record, or file-share URI.

**Source document** is a source asset whose content is treated as a document and can be parsed into text, sections, tables, pages, or other document structures.

**Evidence** is a specific part of the source content that supports a claim or answer. It includes the supporting content and enough location information to find it again.

**Provenance** records where the evidence came from and how it was created or processed. It includes its **lineage**, meaning the ordered chain connecting the evidence to the objects from which it was derived.

**Ground truth** is a fact, label, or expected result that has been validated and accepted as correct for a defined context.

**Reference answer** is an approved way of expressing the ground truth as an answer to a particular question or instruction.

### Example

A company travel-policy PDF is the **source asset** and, after being interpreted as a document, the **source document**.

The sentence stating that claims must be submitted within 30 days is the **evidence**.

The chain:

`PDF → parsed text → paragraph → evidence`

is the evidence **lineage**. The source version, processing steps, and this lineage together form its **provenance**.

The validated rule, “the submission deadline is 30 days,” is the **ground truth**.

The answer, “Employees must submit travel claims within 30 days of completing the journey,” is the **reference answer**.

The resulting question and reference answer can then become a training or evaluation example.
