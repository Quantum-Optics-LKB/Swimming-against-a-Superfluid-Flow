## Superfluid Flow: Self-Propulsion via Vortex-Antivortex Shedding in a Quantum Fluid of Light

### Codes:
- contrast.py contains all image density and phase extraction from raw data
- velocity.py to extract information from fields array
- data_processing.py to run the treatement

### Description of the Dataset
Download the data on Zenodo: [link](futur_link.com)

This dataset contains numerical and experimental data used to generate the figures in our article. The provided files include raw interferograms, processed field data, and computed quantities such as energy spectra and velocity fields.

#### **Figures and Data Sources**

-   **Propagation z-axis scan**: Data from `10181300`
-   **Mach number scan**: Data from `08301817`

#### **Field Data**

The `field.npy` files contain 4D NumPy arrays with shape `(i, j, Ny, Nx)`, where:

-   `i` represents the number of time steps,
    
-   `j` corresponds to the number of images averaged at each time step,
    
-   `Ny, Nx` are the spatial dimensions.
    
-   `field_ref.npy` corresponds to a reference Gaussian field without vortices.
    
-   `field_vortex.npy` contains fluid data with a single vortex, used to measure vortex size at each time step.
    

#### **Computed Quantities**

The dataset also includes various derived quantities, such as energy distributions and velocity fields.

