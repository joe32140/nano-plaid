/* Shared hover-glossary for the SIMD school.
 *
 * Decorates jargon (GEMM, LUT, SDOT, …) in the prose with a subtle dotted
 * underline; hovering (mouse), focusing (keyboard: Tab), or tapping (touch)
 * shows a definition card. One place to edit — the TERMS list below — and it
 * lights up on every class page that loads this file.
 *
 * UX contract:
 *   · discoverable   — dotted underline signals "defined term", not a link
 *   · every device   — mouse hover, keyboard focus, and touch tap all work
 *   · never trapped   — Escape or tap-away dismisses; card never covers itself
 *   · non-destructive — skips code demos, headings, links, and the quiz so it
 *                       can't corrupt the interactive widgets' own DOM
 */
(function () {
  "use strict";

  // term aliases → { name (shown bold), def }. Add entries here.
  const TERMS = [
    { k: ["GEMM"], name: "GEMM — general matrix multiply", def: "The BLAS routine that multiplies two matrices — the workhorse of dense linear algebra, and the 'decompress then multiply' scoring path this course learns to avoid." },
    { k: ["LUT"], name: "LUT — lookup table", def: "A small precomputed array you index instead of computing. Here: the 2ⁿ learned residual corrections a code selects — 16 values for a 4-bit code." },
    { k: ["SIMD"], name: "SIMD — single instruction, multiple data", def: "One CPU instruction that operates on a whole vector of values at once. The basis of NEON, AVX2, and AVX-512." },
    { k: ["BLAS"], name: "BLAS — basic linear algebra subprograms", def: "The standard optimized math library (Apple Accelerate, OpenBLAS, MKL) that provides GEMM and friends." },
    { k: ["MaxSim"], name: "MaxSim — late-interaction score", def: "For each query token take its maximum similarity over all document tokens, then sum. ColBERT's scoring function." },
    { k: ["SDOT", "sdot"], name: "SDOT — NEON int8 dot-product", def: "One ARM instruction doing 16 multiply-adds (four 4-wide dot products) on signed int8 lanes." },
    { k: ["SMMLA", "smmla"], name: "SMMLA — NEON int8 matrix-multiply", def: "ARM's i8mm instruction: a 2×2 tile of int8 dot products — 32 multiply-adds in one instruction, twice SDOT's throughput." },
    { k: ["VNNI"], name: "VNNI — vector neural network instructions", def: "x86 AVX-512's vpdpbusd: a fused unsigned×signed int8 dot-accumulate. SDOT's x86 twin." },
    { k: ["vpdpbusd"], name: "vpdpbusd — AVX-512 VNNI dot", def: "The fused u8×i8 dot-product-accumulate instruction; multiplies a byte vector and accumulates into int32 in one shot." },
    { k: ["AVX2"], name: "AVX2 — x86 256-bit SIMD", def: "32 int8 lanes per register. On essentially every Intel/AMD CPU since ~2013 — the near-universal x86 vector path." },
    { k: ["AVX-512"], name: "AVX-512 — x86 512-bit SIMD", def: "64 int8 lanes per register. On servers (Ice Lake Xeon, Zen 4+) but not universal, so it's runtime-detected." },
    { k: ["NEON"], name: "NEON — ARM 128-bit SIMD", def: "16 int8 lanes per register. On every Apple Silicon and AWS Graviton core." },
    { k: ["pshufb"], name: "pshufb — x86 byte shuffle", def: "Performs 16 table lookups in one instruction — the engine of 4-bit fast-scan. The x86 twin of NEON's tbl." },
    { k: ["tbl"], name: "tbl — NEON table lookup", def: "16 byte-lookups in one instruction; the ARM twin of x86's pshufb. Used to expand residual codes into weights." },
    { k: ["psadbw"], name: "psadbw — x86 sum-of-absolute-differences", def: "Sums |a−b| across bytes in one instruction. Used to fake a masked int8 dot on AVX2 (the SAD trick)." },
    { k: ["SAD"], name: "SAD — sum of absolute differences", def: "Σ|a−b|. With a biased query it computes the masked dot the binary kernel needs, using psadbw." },
    { k: ["popcount"], name: "popcount — population count", def: "The number of set bits in a word, in one instruction. The 'T' side of binary scoring." },
    { k: ["NDCG"], name: "NDCG — normalized discounted cumulative gain", def: "A ranking-quality metric (NDCG@10 here): rewards putting relevant results near the top, normalized to [0,1]." },
    { k: ["i8mm"], name: "i8mm — ARM int8 matrix-multiply extension", def: "The CPU feature that provides the SMMLA instruction; present on Neoverse and newer Apple cores." },
    { k: ["dotprod"], name: "dotprod — ARM dot-product extension", def: "The CPU feature that provides the SDOT instruction; on every modern ARM core." },
    { k: ["Rosetta"], name: "Rosetta — Apple's x86-on-ARM emulator", def: "Runs x86 binaries on Apple Silicon. A stray x86 toolchain builds an emulated binary where NEON silently vanishes — a benchmarking trap." },
    { k: ["Amdahl"], name: "Amdahl's law", def: "Total speedup is capped by the fraction of work you didn't speed up: make 85% of a query 10× faster and the whole query is still only ~2.4× faster." },
    { k: ["transpose-reduce"], name: "transpose-reduce", def: "Folding four query rows' dot products together by transposing them into one register, avoiding a per-row horizontal reduce." },
    { k: ["doc-token-outer"], name: "doc-token-outer loop", def: "Loop order that expands each document token's bits once, then reuses them across all query tokens — amortizing the unpack." },
    { k: ["2P−T", "2P-T"], name: "2P − T — the binary identity", def: "With ±1 document bits, q·s = 2·(Σ q over set bits) − (Σ q). Turns a dot product into a masked integer sum — no multiplies." },
    { k: ["int8"], name: "int8 quantization", def: "Mapping float32 values to 8-bit integers with a per-row scale, so the dot product runs in cheap integer units." },
  ];

  // ── build lookup tables ────────────────────────────────────────────────
  const byWord = new Map(); // exact-case token (prose) → entry
  const byCode = new Map(); // lowercased token (<code>) → entry
  for (const e of TERMS) for (const w of e.k) { byWord.set(w, e); byCode.set(w.toLowerCase(), e); }
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const words = [...byWord.keys()].sort((a, b) => b.length - a.length);
  const RE = new RegExp("\\b(" + words.map(esc).join("|") + ")\\b", "g");

  // Ancestors whose text must not be touched (code demos, headings, links,
  // interactive widgets, the generated quiz).
  const SKIP_TAG = new Set(["A", "CODE", "PRE", "SCRIPT", "STYLE", "NOSCRIPT", "H1", "H2", "H3", "H4", "BUTTON", "SVG", "CANVAS", "INPUT", "TEXTAREA", "SELECT", "LABEL"]);
  const SKIP_SEL = ".eyebrow,.tabs,.course-bar,.widget,.next-class,.gloss,#quiz";

  function skip(node) {
    for (let el = node.parentElement; el; el = el.parentElement) {
      if (SKIP_TAG.has(el.tagName)) return true;
      if (el.matches && el.matches(SKIP_SEL)) return true;
    }
    return false;
  }

  function makeTrigger(text, entry) {
    const s = document.createElement("span");
    s.className = "gloss";
    s.tabIndex = 0;
    s.setAttribute("role", "button");
    s.setAttribute("aria-label", entry.name + ". " + entry.def);
    s.textContent = text;
    s._g = entry;
    return s;
  }

  function decorate(root) {
    // 1) prose text nodes — collect first, then mutate (don't walk new nodes).
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.nodeValue && n.nodeValue.trim() && !skip(n) && RE.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
    });
    const targets = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) targets.push(n);
    for (const node of targets) {
      RE.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0, m;
      while ((m = RE.exec(node.nodeValue))) {
        if (m.index > last) frag.appendChild(document.createTextNode(node.nodeValue.slice(last, m.index)));
        frag.appendChild(makeTrigger(m[0], byWord.get(m[0])));
        last = m.index + m[0].length;
      }
      if (last < node.nodeValue.length) frag.appendChild(document.createTextNode(node.nodeValue.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
    // 2) <code> whose whole text is a term — decorate the element in place
    //    (no text splitting, so code formatting is untouched).
    for (const c of root.querySelectorAll("code")) {
      if (c.closest("a,.gloss")) continue;
      const entry = byCode.get(c.textContent.trim().toLowerCase());
      if (entry) {
        c.classList.add("gloss");
        c.tabIndex = 0;
        c.setAttribute("role", "button");
        c.setAttribute("aria-label", entry.name + ". " + entry.def);
        c._g = entry;
      }
    }
  }

  // ── tooltip ────────────────────────────────────────────────────────────
  const tip = document.createElement("div");
  tip.className = "gloss-tip";
  tip.setAttribute("role", "tooltip");
  let current = null;

  function place(el) {
    const r = el.getBoundingClientRect();
    tip.style.maxWidth = Math.min(320, window.innerWidth - 24) + "px";
    const tr = tip.getBoundingClientRect();
    const above = r.top - tr.height - 10 >= 8;
    let top = (above ? r.top - tr.height - 10 : r.bottom + 10) + window.scrollY;
    let left = r.left + r.width / 2 - tr.width / 2 + window.scrollX;
    left = Math.max(window.scrollX + 8, Math.min(left, window.scrollX + window.innerWidth - tr.width - 8));
    tip.style.top = top + "px";
    tip.style.left = left + "px";
  }
  function show(el) {
    if (!el._g) return;
    current = el;
    tip.innerHTML = "";
    const b = document.createElement("b");
    b.textContent = el._g.name;
    tip.appendChild(b);
    tip.appendChild(document.createTextNode(el._g.def));
    tip.classList.add("show");
    place(el);
  }
  function hide() { current = null; tip.classList.remove("show"); }

  function init() {
    document.body.appendChild(tip);
    decorate(document.body);

    const fine = window.matchMedia("(hover:hover)").matches;
    if (fine) {
      document.addEventListener("pointerover", (e) => { const g = e.target.closest(".gloss"); if (g) show(g); });
      document.addEventListener("pointerout", (e) => { const g = e.target.closest(".gloss"); if (g && !g.contains(e.relatedTarget)) hide(); });
    }
    document.addEventListener("focusin", (e) => { const g = e.target.closest(".gloss"); if (g) show(g); });
    document.addEventListener("focusout", (e) => { if (e.target.closest(".gloss")) hide(); });
    // Tap: toggle on the term, dismiss on tap-away (covers touch + click).
    document.addEventListener("click", (e) => {
      const g = e.target.closest(".gloss");
      if (g) { current === g ? hide() : show(g); }
      else if (current) hide();
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
    window.addEventListener("scroll", () => { if (current) place(current); }, { passive: true });
    window.addEventListener("resize", () => { if (current) place(current); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
