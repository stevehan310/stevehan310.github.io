# Change Log

Personal site update log (not to be confused with `CHANGELOG.md`, which tracks the upstream Minimal Mistakes theme).

## 2026-08-02

- Added new post: "Basic Neural Network from Scratch: NumPy vs PyTorch" (`_posts/2026-08-02-Neural_Network_From_Scratch_with_NumPy_and_PyTorch.md`)
  - Single-hidden-layer NN (2→6→1, tanh/sigmoid) trained on a non-linearly-separable XOR-pattern dataset
  - Forward/backward propagation implemented and executed both by hand in NumPy and via `autograd` in PyTorch
  - Companion notebook: `notebooks/2026-08-02-Neural_Network_From_Scratch_with_NumPy_and_PyTorch.ipynb`
  - Generated plots stored under `assets/images/NN_Scratch/`
- Added MathJax to `_includes/head/custom.html` so kramdown-rendered math (`$$...$$` block, `$...$` inline) displays site-wide
  - Initial config only registered `\(...\)` as an inline delimiter; fixed to also register `$...$`, since kramdown leaves inline math in that form untouched
- Translated the new post/notebook's explanatory text and code comments from Korean to English (content and generated outputs unchanged)
- Added `PROJECT_NOTES.md` documenting the repo's directory structure and posting workflow
