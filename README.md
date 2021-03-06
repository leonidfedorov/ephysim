# ephysim
**Simulation of Electrophysiological Signals.**

Great deal of work has been done on the membrane potential theory and computation with details of neuronal morphology, distributions of ion channels, population connectivity and even plasticity. Less, but still formidable attention has been paid to formalize methods for spike sorting, field potential and multi-unit activity analyses. Here we want to marry the two: similarly in spirit to [1], in the simplest possible manner this project models the extracellular voltage time series using physical approximations of electric field at a point and HH-model-based stochastic differential equations. Special attention is paid to being able to play with channel configurations in space and the distance of this point set to the membrane. The goal is to support new geometry-aware and biophysically informed data analytic methods.

[1] Carl Gold, Darrell A. Henze, Christof Koch, and György Buzsáki _On the Origin of the Extracellular Action Potential Waveform: A Modeling Study_ Journal of Neurophysiology Vol. 95, No. 5
