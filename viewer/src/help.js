// The words of the page's two dialogs: the keys and the pointer, and the work the page is built on.

/** The keys main.js acts on, and what the pointer does; the keys dialog lists them in this order. */
export const KEYS = [
  { group: 'Pointer' },
  { keys: 'drag', what: 'Turn the 3D view' },
  { keys: 'right-drag', what: 'Pan' },
  { keys: 'wheel', what: 'Zoom' },
  { keys: 'click', what: 'Follow a region, from the cloud, the scene graph or the strip' },
  { group: 'Frames' },
  { keys: '← →', what: 'Previous or next frame' },
  { keys: 'Space', what: 'Play or pause' },
  { group: 'View' },
  { keys: 'W', what: 'World mode: the clip\'s frames stacked in one space' },
  { keys: 'G', what: 'The scene graph drawn on the cloud' },
  { keys: 'S', what: 'The shown track\'s regions painted on the cloud' },
  { keys: 'I', what: 'Every point, the tissue only, or the instruments only' },
  { keys: 'O', what: 'World mode: the followed region alone' },
  { keys: 'P', what: 'World mode: each frame\'s image under its regions in the strip' },
  { keys: 'R', what: 'Reset the 3D view' },
  { keys: 'Esc', what: 'Stop following the region' },
  { keys: '?', what: 'This list' },
];

/** The data and models the shown clips were made from, each with what it is used for and its reference. */
export const CREDITS = [
  {
    group: 'Data',
    items: [
      { name: 'Cholec80', use: 'the frames of the CholecSeg8k clips',
        cite: 'A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy. EndoNet: A Deep '
          + 'Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Transactions on Medical Imaging 36(1):86–97, 2017.',
        url: 'https://doi.org/10.1109/TMI.2016.2593957' },
      { name: 'CholecSeg8k', use: 'the manual annotation of the Cholec80 frames',
        cite: 'W.-Y. Hong, C.-L. Kao, Y.-H. Kuo, J.-R. Wang, W.-L. Chang, C.-S. Shih. CholecSeg8k: A Semantic '
          + 'Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80. arXiv:2012.12453, 2020.',
        url: 'https://arxiv.org/abs/2012.12453' },
      { name: 'ATLAS-120k', use: 'the ATLAS-120k clips and their manual annotation',
        cite: 'R. L. P. D. de Jong, T. J. M. Jaspers, R. A. H. Vervoort, et al. Surgical Anatomy Recognition with '
          + 'Context Learning using Foundation Representations. arXiv:2606.22124, 2026.',
        url: 'https://arxiv.org/abs/2606.22124' },
    ],
  },
  {
    group: 'Depth and camera pose',
    items: [
      { name: 'Depth Anything 3', use: 'depth and camera pose, on which the scene graphs are built',
        cite: 'H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, B. Kang. Depth Anything 3: '
          + 'Recovering the Visual Space from Any Views. arXiv:2511.10647, 2025.',
        url: 'https://arxiv.org/abs/2511.10647' },
      { name: 'π³ (Pi3X)', use: 'the point clouds of the clips drawn from Pi3X',
        cite: 'Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, T. He. π³: '
          + 'Permutation-Equivariant Visual Geometry Learning. arXiv:2507.13347, 2025.',
        url: 'https://arxiv.org/abs/2507.13347' },
    ],
  },
  {
    group: 'Segmentation and tracking',
    items: [
      { name: 'SAM', use: 'the zero-shot regions of the seed frame',
        cite: 'A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, '
          + 'W.-Y. Lo, P. Dollár, R. Girshick. Segment Anything. ICCV 2023, pp. 3992–4003.',
        url: 'https://doi.org/10.1109/ICCV51070.2023.00371' },
      { name: 'SAM 2', use: 'the memory-based video tracking SAM 3 builds on',
        cite: 'N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, '
          + 'et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024.',
        url: 'https://arxiv.org/abs/2408.00714' },
      { name: 'SAM 3', use: 'the video tracker that carries each region through the clip',
        cite: 'N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, et al. SAM 3: Segment '
          + 'Anything with Concepts. arXiv:2511.16719, 2025.',
        url: 'https://arxiv.org/abs/2511.16719' },
    ],
  },
];
