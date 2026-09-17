MIT License

Copyright (c) 2026 Karn Jadhav

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

---

## Scope of this license

This MIT license covers the **source code** in this repository only:
FastAPI service code, training/evaluation/inference scripts, dataset
preparation/validation tooling, the Node.js backend, and the React frontend.

**This license does NOT cover:**

- **Training datasets** (RDD2022, Pothole-600, Attain, or any other
  third-party dataset used to prepare `datasets/potholes/`). Each retains
  its own original license. See `DATASETS.md` for the current status of
  each source, including licenses that are not yet confirmed.
- **Trained model weights** (`best.pt`, `best.onnx`, or any checkpoint
  derived from training on the above datasets). A model's redistribution
  terms are constrained by the MOST RESTRICTIVE license among the datasets
  used to train it — an MIT license on this code does not override that.
  Do not treat weight files as freely licensed just because the code that
  produced them is.

If any dataset license listed in `DATASETS.md` turns out to be
non-commercial-only, share-alike, or otherwise restrictive, that
restriction applies to models trained on it and to any product built on
those models — regardless of this file.