#!/usr/bin/env python3
"""Turn page_curve_amps_R1.pdf (written by page_curve_amps_R1.py) into the paper's
Figure 4, page_curve_amps.pdf, by the two text edits made after the run:
  1. the panel (a) title "(a) Conditional Page curve" becomes "(a) Page curve";
  2. the peak label "0.4773, 0.5755" of the dashed f = 0.887 curve is removed,
     because the caption does not discuss that pair.
No curve, axis or number other than that label changes.
Requires PyMuPDF (import fitz).
Usage: python page_curve_amps_postedit.py page_curve_amps_R1.pdf page_curve_amps.pdf
"""
import sys
import fitz

src, out = sys.argv[1], sys.argv[2]
doc = fitz.open(src)
page = doc[0]

# 1. panel (a) title
hits = page.search_for("(a) Conditional Page curve")
assert len(hits) == 1, hits
r = hits[0]
fs = [s["size"] for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
      for s in l.get("spans", []) if "Conditional Page curve" in s["text"]][0]
page.add_redact_annot(r, fill=(1, 1, 1))
page.apply_redactions()
page.insert_text((r.x0, r.y1 - 0.2 * fs), "(a) Page curve", fontsize=fs,
                 fontname="helv", color=(0, 0, 0))

# 2. drop the f = 0.887 peak label
for t in ("0.4773, 0.5755", "0.4773", "0.5755"):
    hits = page.search_for(t)
    if hits:
        break
assert hits, "peak label not found"
for r in hits:
    page.add_redact_annot(r)
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                      graphics=fitz.PDF_REDACT_LINE_ART_NONE)
doc.save(out, garbage=3, deflate=True)
