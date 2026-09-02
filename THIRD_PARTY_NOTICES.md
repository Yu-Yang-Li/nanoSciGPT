# Third-party notices

## nanoGPT

Parts of the minimal GPT organization and teaching flow are adapted from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), distributed under the
MIT License.

```text
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Technical references not copied into this repository

- `nanochat` informs the single-command, inspectable-artifact workflow.
- `prot-gpt` informs independent variable-length sequence handling.
- `nanoGPT-DNA` informs the nucleotide-level classroom boundary.
- `EarthPT` informs the documented boundary between discrete tokens and
  continuous scientific observations.
- `AstroPT`, `GPTCast`, `CGCNN`, and `SpectralGPT` inform the documented
  boundaries for image patches, weather grids, periodic graphs, and spectra.

The latter projects are technical references only. Their source code is not
vendored into nanoSciGPT.

## Bundled teaching data

- Protein records come from UniProtKB. UniProt applies CC BY 4.0 to
  copyrightable database content and requests attribution; other rights may
  still apply. See <https://www.uniprot.org/help/license/>.
- The DNA excerpt comes from the UCSC hg38 download. UCSC asks users to
  acknowledge data contributors and to review source-specific restrictions.
  See <https://genome.ucsc.edu/goldenPath/help/>.
- The ESOL table is the Delaney solubility dataset as distributed by DeepChem.
  Cite the original work and do not infer a broader dataset license from
  DeepChem's code license.
- The text warm-up uses a public-domain Shakespeare corpus mirrored by the
  char-rnn project.
- The supervised-learning opening uses the Palmer Penguins measurements
  collected by Kristen Gorman and the Palmer Station Antarctica LTER. The
  simplified dataset is distributed by the `palmerpenguins` project under
  CC0-1.0. This repository retains the 342 rows with complete bill, flipper,
  and body-mass measurements and cites the original ecology study:
  Gorman, Williams, and Fraser (2014), PLOS ONE 9(3): e90081.
- The current scientific-regression opening uses a fixed 2000-spectrum subset
  prepared from the ATLAS-A directly observed stellar library distributed by
  the National Astronomical Data Center of China. Each spectrum is normalized
  and rebinned to 128 flux features; `Teff_sed` is retained as `teff`. The data
  page provides public download and the citation Ji et al. (2023), DOI
  `10.12149/101205`; this repository does not infer a broader open-data license.

Exact URLs and bundled files are recorded in `data/manifest.json`.

The six structured fixtures (`weather`, `crystal`, `structure3d`, `image`,
`spectrum`, and `field`) are generated locally by nanoSciGPT. They do not
contain third-party observational or simulation data.
