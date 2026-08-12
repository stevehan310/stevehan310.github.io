# Change Log

Personal site update log (not to be confused with `CHANGELOG.md`, which tracks the upstream Minimal Mistakes theme).

## 2026-08-11

- Added new post: "Mastering the Adam Optimizer: From Concept to NumPy & PyTorch Implementation" (`_posts/2026-08-11-Adam_Optimizer_From_Scratch_NumPy_vs_PyTorch.md`)
  - Covers the Adam algorithm (first/second moments, bias correction, epsilon), implemented from scratch in NumPy and again via `torch.optim.Adam`
  - Published from a standalone markdown draft (`_md/adam-optimizer-en.md`); draft folder removed after publishing
  - Companion notebook added afterward: `notebooks/2026-08-11-Adam_Optimizer_From_Scratch_NumPy_vs_PyTorch.ipynb`, executed with the `ai_agent` conda env (numpy 2.2.4, torch 2.7.1) so cell outputs are real, not transcribed

## 2026-08-10

- Added new post: "Implementing Gradient Descent From Scratch: NumPy vs PyTorch" (`_posts/2026-08-10-Gradient_Descent_From_Scratch_NumPy_vs_PyTorch.md`)
  - Covers linear regression gradient descent (manual gradient + autograd) and batch/stochastic/mini-batch variants, both in NumPy and PyTorch
  - Published directly from a standalone markdown draft (`md/gradient_descent_blog_post.md`); no companion notebook exists for this post
  - Removed the redundant top-level `#` heading duplicating the frontmatter `title` — adopted as the convention for new posts going forward (see 2026-08-11 post above, which follows it from the start)

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
