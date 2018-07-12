# fbpicker
Deep Neural Network and Multi-pattern Based Algorithm for Picking First-arrival Traveltimes

Open-source deep learning library was applied to create generative model for automatic estimation of first-break times for land vibroseis dataset. The core mechanism relies on pattern recognition techniques and signal-based methods of generating these patterns. The first-break picking is treated here as a binary classification problem that requires a model to differentiate first-break sample from all others samples. In order to provide a sufficient training dataset  a STA/LTA method, an entropy-based method, and a variogram fractal-dimension method have been used. The approach appears robust and flexible in a way of adding new pattern generators that might contribute to even better performance.
Moreover, already trained models can be saved and reproduced for another dataset collected with similar acquisition parameters (e.g., in multi-line surveys).
