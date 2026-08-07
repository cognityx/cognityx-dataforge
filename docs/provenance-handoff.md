# Provenance handoff

## Where it fits

Ingest keeps the source facts. DataForge creates learning material from those
facts. The v3.2 handoff connects the two without making DataForge parse a source
file again:

```text
source file
   -> Ingest canonical content, source relationships, and exact addresses
   -> DataForge paragraph questions or composite Knowledge Units
   -> dataset publication
```

Canonical content is the one Ingest-owned copy of extracted source text. A
Source Graph is a small map of resources, sections, nodes, and explicit
relationships that Ingest can prove. A provenance address is a stable pointer to
exact evidence in that map. These records let DataForge answer “what supports
this output?” without reopening the PDF or understanding a parser's private
format.

## What Ingest produces

Each successful v3.2 document has:

- `canonical-content.json`, containing parser-neutral source content;
- `source-graph.json`, containing source structure and explicit relationships;
- `provenance-addresses.json`, containing strong evidence pointers; and
- provenance v2, containing logical `storage://` locations for those artifacts.

The Ingest run manifest may also include `dataforge_source_refs`. Each item is a
compact packing list with the document ID and the four relevant Storage
locations. DataForge still reads provenance v2 and requires exact agreement. It
does not choose between contradictory references.

Old provenance v1 remains valid for the existing paragraph and Knowledge Unit
recipes. It is not described as v3.2-capable when the canonical, graph, and
address artifacts are absent. A caller sees a typed “v3.2 handoff unavailable”
failure only when it asks for the new path.

## What DataForge is allowed to store

DataForge may briefly reconstruct source text in memory to call an approved
generator. For example, a paragraph segment contains a node ID, so DataForge can
read that node's canonical text long enough to produce a question and answer.
The reconstructed paragraph is then discarded.

A persisted handoff may contain generated:

- questions;
- answers; and
- claim text.

It may also contain IDs for the graph revision, segmentation view, segment, node
spans, support addresses, evidence set, and allowed or excluded relations.

It may not contain another independent `source_text`, `paragraph_text`, excerpt,
quoted context, or canonical-text copy. This is a structural rule about fields,
not a wording test. A correct generated answer may legitimately repeat words
from its source.

## Paragraph questions and exact support

A paragraph view is a list of references into canonical content. DataForge first
proves that the view is bound to those exact canonical bytes. It then locates the
requested segment, reconstructs its node spans in memory, and finds strong
addresses for exactly those spans.

Every address passes through Ingest's resolver. Only status `exact` can support a
handoff. `ambiguous`, `obsolete`, `forbidden`, `unresolved`, and redirected
results fail before the generator runs. The generator receives text only; it is
never asked to invent support IDs.

## Composite Knowledge Units

A composite Knowledge Unit combines claims that need more than one exact source
passage. The immediate v3.2 path begins with an explicit evidence-set address.
An evidence-set address pairs each caller-owned claim ID with one strong address,
in order. It is intent supplied by a business task or DataForge, not something
Ingest guesses from similar wording.

DataForge can consume an evidence set already present in the address catalog. It
can also validate an explicit DataForge-owned intent against the catalog's strong
addresses in memory. This does not modify the immutable Ingest artifact or
pretend that Ingest emitted the derived intent.

The complete evidence set must resolve exactly before claim generation. For each
member, DataForge reconstructs the addressed node or division content only for
that call. The generator returns claim text only. Deterministic code attaches the
already-proven claim ID and support address.

## Gold relationship closure

Composite work can follow explicit Source Graph relationships from a seed
section. This is a bounded relationship closure: start at the section, inspect
its direct nodes, and continue through accepted concrete targets while remembering
visited IDs so cycles terminate.

Only relationships returned as gold-safe by `SourceGraph.outgoing(...,
gold_only=True)` are followed. An edge is still recorded when it returns to an
already visited target, but a candidate target on an ambiguous edge is never
traversed. Allowed and excluded relation IDs are emitted in the Source Graph's
stable order.

This is not a semantic knowledge graph. T09 does not extract entities, infer new
meaning, add embeddings, use a vector database, rank retrieval results, or run
GraphRAG.

## Why parser payloads are unnecessary

The handoff loads only canonical content, the Source Graph, and provenance
addresses through public Ingest readers. It never requests
`parser/{backend}.json`, a T01 native payload, or the original PDF. No source
reparse or parser execution occurs. This also means a safely purged parser-native
payload does not invalidate canonical strong support.

## Storage and repeatability

Handoffs serialize to deterministic JSON and are written through the configured
DataForge Storage role. Retrying the same identity with identical bytes succeeds.
Retrying that identity with changed bytes fails rather than overwriting the first
artifact. The handoff contains logical Storage identity only and accepts no local
storage root argument.

## Current limit and next boundary

A cross-document Knowledge Unit needs one Source Graph that already contains all
participating resources and their explicit cross-resource relationships. T09 does
not merge unrelated per-document graphs or invent edges to make a composite work.

T09 adds no CLI or SDK command. Existing `paragraph-qa`, `knowledge-unit-qa`, and
`knowledge-unit-probed-qa` recipes keep their current meaning. SDK and CLI
surfaces belong to T10.
