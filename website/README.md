# CMA-ES-IG project page

A static site with an interactive demo: rank Jaco2 handovers and watch CMA-ES-IG adapt.
Everything runs client-side, so it can be hosted on GitHub Pages as-is.

```
website/
├── index.html              # project page (paper summary, figures, BibTeX, demo)
├── static/
│   ├── js/cmaesig.js       # CMA-ES, CMA-ES-IG, Infogain, particle belief
│   ├── js/demo.js          # three.js scene + ranking UI
│   ├── css/style.css
│   ├── data/handovers.f32  # 1000 x 50 x 6 Jaco2 joint trajectories (float32, from applications/robot_handover/handovers.npy)
│   ├── data/embeddings.f32 # 1000 x 8 autoencoder features (float32, from handover_embeddings.npy)
│   ├── robot/j2n6s300/     # Kinova j2n6s300 URDF from kinova-ros (BSD-3, see LICENSE); meshes simplified to ~15% of the
│   │                       #   original triangles and converted from .dae to .glb to keep the demo light
│   ├── images/             # figures extracted from the paper
│   └── cma-es-ig.pdf
```

## Preview locally

ES modules don't load from `file://`, so serve the folder:

```bash
cd website && python -m http.server 8000   # open http://localhost:8000
```

## Deploy to GitHub Pages

`.github/workflows/pages.yml` uploads `website/` on every push to `main` that touches it.
Enable it once under **Settings → Pages → Build and deployment → Source: GitHub Actions**.
(Alternatively, copy the contents of `website/` to the root of a `gh-pages` branch.)

three.js and urdf-loader are loaded from jsDelivr via the import map in `index.html`.

## Notes

- Joint angles in `handovers.npy` use the Kinova firmware convention (home = `[4.8, 2.9, 1.0, 4.2, 1.4, 1.3]` rad),
  which matches the kinova-ros `j2n6s300` URDF directly.
- Features are standardised as `(e - mean) / (2 * std)` so the dataset roughly fills the `[-1, 1]` box CMA-ES searches.
  As in the user study, K = 3 candidates per round with a CMA-ES population size of 3. CMA-ES-IG draws D = 100 samples; σ₀ = 0.5.
