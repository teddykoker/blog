---
layout: page
---

<!-- <img alt="Teddy Koker" src="/images/profile.jpg"
    style="float: right; max-width: 33%; margin: 0 0 1em 2em; border-radius: 50%"> -->

# Teddy Koker

I am a PhD student at [MIT EECS](https://www.eecs.mit.edu/), advised by Professor [Tess Smidt](https://blondegeek.github.io/).

Previously, I spent several years as a technical staff member at MIT Lincoln Laboratory, and performed research at Lightning AI and Harvard Medical School.

[Email](mailto:teddy.koker@gmail.com) / [CV](/koker_cv.pdf) / [Github](https://github.com/teddykoker) / [Google Scholar](https://scholar.google.com/citations?user=br990A8AAAAJ) / [Twitter](https://twitter.com/teddykoker)

<div style="margin: 1em 0;">
  <div id="molecule-viewer" style="width: 100%; height: 350px;"></div>
  <br>
  <small>Dissolution of a NaCl nanocrystal in water, simulated with the
   <a href="https://arxiv.org/abs/2508.16067">Nequix MP</a> interatomic potential (<a href="https://github.com/teddykoker/nequix-examples/tree/main/nacl-dissolution#readme">details</a>).
  </small>
  <div>
  <button id="play-btn">Play</button>
  <small id="frame-info"></small>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.183.0/build/three.module.min.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.183.0/examples/jsm/"
  }
}
</script>
<script type="module">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

  const COLORS = { H: 0xffffff, Na: 0xab5cf2, Cl: 0x1ff01f, O: 0xff0d0d, C: 0x909090, N: 0x3050f8 };
  const RADII = { H: 0.12, Na: 1.1, Cl: 0.95, O: 0.2, C: 0.5, N: 0.45 };
  const wrap = (x, L) => x - L * Math.floor(x / L);

  function toAtoms(f) {
    const [lx, ly, lz] = [f.cell[0], f.cell[1], f.cell[2]];
    const pbc = f.pbc ?? [true, true, true];
    return f.positions.map((p, i) => {
      let x = pbc[0] ? wrap(p[0], lx) : p[0];
      let y = pbc[1] ? wrap(p[1], ly) : p[1];
      let z = pbc[2] ? wrap(p[2], lz) : p[2];
      return { element: f.symbols[i] ?? 'X', x: x - lx / 2, y: y - ly / 2, z: z - lz / 2 };
    });
  }

  const container = document.getElementById('molecule-viewer');
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 10000);
  camera.position.set(35, 5, 0);

  const scene = new THREE.Scene();
  scene.background = null;

  const ambient = new THREE.AmbientLight(0x404040, 1);
  const directional = new THREE.DirectionalLight(0xffffff, 1);
  scene.add(ambient, directional, directional.target);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  let trajectoryFrames = [];
  let molecule = null;
  let meshRefs = [];
  let isPlaying = false;
  let frameIdx = 0;
  const FPS = 30;

  function createMolecule(atoms) {
    const group = new THREE.Group();
    const refs = [];
    for (const a of atoms) {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(RADII[a.element], 16, 16),
        new THREE.MeshPhongMaterial({ color: COLORS[a.element], shininess: 30, specular: 0x222222 })
      );
      m.position.set(a.x, a.y, a.z);
      group.add(m);
      refs.push(m);
    }
    return { group, refs };
  }

  function applyFrame(frame) {
    if (!frame) return;
    if (frame.length !== meshRefs.length) {
      if (molecule) scene.remove(molecule);
      const { group, refs } = createMolecule(frame);
      molecule = group;
      meshRefs = refs;
      scene.add(molecule);
    } else {
      for (let i = 0; i < frame.length; i++) meshRefs[i].position.set(frame[i].x, frame[i].y, frame[i].z);
    }
  }

  const playBtn = document.getElementById('play-btn');
  const frameInfo = document.getElementById('frame-info');

  const loadJsonGz = async url => JSON.parse(await new Response((await fetch(url)).body.pipeThrough(new DecompressionStream('gzip'))).text());

  // NOTE: convert from .xyz to json.gz with _scripts/xyz_to_jsongz.py
  loadJsonGz('/images/nacl_h2o_opt.json.gz')
    .then(data => {
      const frames = (Array.isArray(data) ? data : []).map(toAtoms);
      if (frames[0]) {
        const { group, refs } = createMolecule(frames[0]);
        molecule = group;
        meshRefs = refs;
        scene.add(molecule);
      }
    })
    .catch(err => console.error('Failed to load structure:', err));

  playBtn.addEventListener('click', async () => {
    if (isPlaying) { isPlaying = false; playBtn.textContent = 'Play'; return; }
    if (trajectoryFrames.length > 0) { isPlaying = true; playBtn.textContent = 'Pause'; return; }
    playBtn.disabled = true;
    frameInfo.textContent = 'Loading trajectory…';
    try {
      // NOTE: convert from .xyz to json.gz with _scripts/xyz_to_jsongz.py
      const data = await loadJsonGz('/images/nvt.json.gz');
      trajectoryFrames = (Array.isArray(data) ? data : []).map(toAtoms);
      frameInfo.textContent = `Loaded ${trajectoryFrames.length} frames`;
      if (trajectoryFrames.length > 0) { isPlaying = true; playBtn.textContent = 'Pause'; }
    } catch (err) { console.error(err); frameInfo.textContent = 'Failed to load trajectory'; }
    playBtn.disabled = false;
  });

  let lastTime = performance.now();
  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    if (isPlaying && trajectoryFrames.length > 0) {
      frameIdx = (frameIdx + (now - lastTime) / 1000 * FPS) % trajectoryFrames.length;
      applyFrame(trajectoryFrames[Math.floor(frameIdx)]);
      frameInfo.textContent = `Frame ${Math.floor(frameIdx) + 1} / ${trajectoryFrames.length}`;
    }
    lastTime = now;
    controls.update();
    directional.position.copy(camera.position);
    directional.target.position.copy(controls.target);
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);

  new ResizeObserver(() => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }).observe(container);
</script>

## Research

I am broadly interested in machine learning and its applications to the computational sciences. My most recent work focuses on deep learning within the field of computational chemistry and materials science.

<br>
*PFT: Phonon fine-tuning for Machine Learned Interatomic Potentials*<br>
**Teddy Koker**, Abhijeet Gangan, Mit Kotak, Jaime Marian, Tess Smidt.<br>
Preprint.<br>
[arXiv](https://arxiv.org/abs/2601.07742) / [Code](https://github.com/atomicarchitects/nequix) / [Blog](/2026/02/pft/)

<br>
*Training a Foundation Model for Materials on a Budget*<br>
**Teddy Koker**, Mit Kotak, Tess Smidt.<br>
NeurIPS AI for Accelerated Materials Design Workshop, 2025.<br>
[arXiv](https://arxiv.org/abs/2508.16067) / [Code](https://github.com/atomicarchitects/nequix)

<br>
*Higher-Order Equivariant Neural Networks
for Charge Density Prediction in Materials*<br>
**Teddy Koker**, Keegan Quigley, Eric Taw, Kevin Tibbetts, Lin Li.<br>
npj Computational Materials, 2024. Also at NeurIPS AI for Science Workshop, 2023.<br>
[Publication](https://www.nature.com/articles/s41524-024-01343-1) / [arXiv](https://arxiv.org/abs/2312.05388) / [Code](https://github.com/AIforGreatGood/charge3net)

<br>
*UniTS: Building a Unified Time Series Model*<br>
Shanghua Gao, **Teddy Koker**, Owen Queen, Thomas Hartvigsen, Theodoros Tsiligkaridis, Marinka Zitnik.<br>
NeurIPS, 2024.<br>
[arXiv](https://arxiv.org/abs/2403.00131) / [Code](https://github.com/mims-harvard/UniTS) / [Website](https://zitniklab.hms.harvard.edu/projects/UniTS/)

<br>
*Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency*<br>
Owen Queen, Thomas Hartvigsen, **Teddy Koker**, Huan He, Theodoros Tsiligkaridis, Marinka Zitnik.<br>
NeurIPS, 2023 (spotlight).<br>
[arXiv](https://arxiv.org/abs/2306.02109) / [Code](https://github.com/mims-harvard/TimeX) / [Website](https://zitniklab.hms.harvard.edu/projects/TimeX/)

<br>
*Domain Adaptation for Time Series Under Feature and Label Shifts*<br>
Huan He, Owen Queen, **Teddy Koker**, Consuelo Cuevas, Theodoros Tsiligkaridis, Marinka Zitnik.<br>
International Conference on Machine Learning (ICML), 2023.<br>
[arXiv](https://arxiv.org/abs/2302.03133) / [Code](https://github.com/mims-harvard/Raincoat) / [Website](https://zitniklab.hms.harvard.edu/projects/Raincoat/)

<br>
*Graph Contrastive Learning for Materials*<br>
**Teddy Koker**, Keegan Quigley, Will Spaeth, Nathan C. Frey, Lin Li.<br>
NeurIPS AI for Accelerated Materials Design Workshop, 2022.<br>
[arXiv](https://arxiv.org/abs/2211.13408)

<br>
*AAVAE: Augmentation-Augmented Variational Autoencoders.*<br>
William Falcon, Ananya Harsh Jha, **Teddy Koker**, Kyunghyun Cho.<br>
Preprint.<br>
[arXiv](https://arxiv.org/abs/2107.12329) / [Code](https://github.com/gridai-labs/aavae)

<br>
*TorchMetrics: Measuring Reproducibility in PyTorch*<br>
N. Detlefsen, J. Borovec, J. Schock, A. Jha, **T. Koker**, L. Liello, D. Stancl, C. Quan, M. Grechkin, W. Falcon.<br>
The Journal of Open Source Software, 2022.<br>
[Publication](https://joss.theoj.org/papers/10.21105/joss.04101) / [Code](https://github.com/Lightning-AI/metrics)


<br>
*U-Noise: Learnable Noise Masks for Interpretable Image Segmentation.*<br>
**T. Koker**, F. Mireshghallah, T. Titcombe, G. Kaissis.<br>
International Conference on Image Processing (ICIP), 2021.<br>
[Publication](https://ieeexplore.ieee.org/document/9506345) / [arXiv](https://arxiv.org/abs/2101.05791) / [Code](https://github.com/teddykoker/u-noise)

<br>
*On Identification and Retrieval of Near-Duplicate Biological Images: A New Dataset and Protocol.*<br>
**T. Koker**\*, S. S. Chintapalli\*, S. Wang, B. A. Talbot, D. Wainstock, M. Cicconet, M. C. Walsh.<br>
International Conference on Pattern Recognition (ICPR), 2020.<br>
[Publication](https://ieeexplore.ieee.org/document/9412849) / [PDF](/docs/binder.pdf) / [Code](https://github.com/HMS-IDAC/BINDER)


<br>
*Cryptocurrency Trading Using Machine Learning.*<br>
**Teddy Koker**, Dimitrios Koutmos.<br>
Journal of Risk and Financial Management, 2020.<br>
[Publication](https://www.mdpi.com/1911-8074/13/8/178)
