import json
from dataclasses import dataclass
from numpy import pi, tan

@dataclass
class LinearHomo1D:
    '''based on K-Wave example 'example_na_modelling_nonlinearity.m' '''
    with open('params.json', 'r') as f:
        params_json = json.load(f)['LinearHomo1D']

    # parameters
    p0: float = params_json['p0']                                              # source pressure [Pa]
    c0: float = params_json['c0']                                              # sound speed [m/s]
    rho0: float = params_json['rho0']                                          # density [kg/m^3]
    alpha_0: float = params_json['alpha_0']                                    # absorption coefficient [dB/(MHz^2 cm)]
    sigma: float = params_json['sigma']                                        # shock parameter
    source_freq: float = params_json['source_freq']                            # frequency [Hz]
    points_per_wavelength: int = params_json['points_per_wavelength']          # number of grid points per wavelength at f0 [grid point]
    wavelength_separation: int = params_json['wavelength_separation']          # separation (number of waves) between the source and detector
    pml_size: int = params_json['pml_size']                                    # PML size   
    pml_alpha: float = params_json['pml_alpha']                                # PML absorption coefficient [Np/grid point]
    CFL: float = params_json['CFL']                                            # CFL number        

    # grid
    dx: float = c0 / (points_per_wavelength * source_freq) 
    Nx: int = wavelength_separation * points_per_wavelength + 20               # number of grid points in the x direction (distance + 20) [grid point]

    # medium
    sound_speed: float = c0
    density: float = rho0
    alpha_power: int = 2
    alpha_coeff: float = alpha_0
    f_max: float = sound_speed / (2 * dx) 
    
    # source
    source_pos: int = 10                                                        # source position [grid point]
    x0: float = (source_pos-1) * dx                                             # source position [m]
    
    # sensor
    x_px: int = wavelength_separation * points_per_wavelength                   # number of grid points between the source and detector [grid point]
    x: float = x_px * dx                                                        # distance between the source and detector [m]
    x_end: float = (x_px + 20) * dx                                             # end of the grid in the x direction [m]

    detector_pos: int = source_pos + x_px                                       # position of the detector [grid point]
    x_detector: float = (detector_pos-1) * dx                                   # position of the detector [m]

    # nonlinearity
    k: float = 2 * pi * source_freq / c0
    BonA: float = 2 * (sigma / (p0 / (rho0 * c0**2) * k * x) - 1)
    tau: float = 0.0
    eta: float = 0.0
    d: float = 0.0
    y: float = 1.0
    
    # time
    dt: float = 1 / (round(points_per_wavelength / CFL) * source_freq)
    t_end: float = 25e-6
    Nt: int = round(t_end / dt)
    # print(f"dt: {dt}, dx: {dx}, rho0: {rho0}, c0:{c0}, p0:{p0}, f:{source_freq}")

class LinearHomo1D_set2:
    '''based on K-Wave example 'example_na_modelling_nonlinearity.m' '''
    with open('params.json', 'r') as f:
        params_json = json.load(f)['LinearHomo1D_2']

    # parameters
    p0: float = params_json['p0']                                              # source pressure [Pa]
    c0: float = params_json['c0']                                              # sound speed [m/s]
    rho0: float = params_json['rho0']                                          # density [kg/m^3]
    alpha_0: float = params_json['alpha_0']                                    # absorption coefficient [dB/(MHz^2 cm)]
    sigma: float = params_json['sigma']                                        # shock parameter
    source_freq: float = params_json['source_freq']                            # frequency [Hz]
    points_per_wavelength: int = params_json['points_per_wavelength']          # number of grid points per wavelength at f0 [grid point]
    wavelength_separation: int = params_json['wavelength_separation']          # separation (number of waves) between the source and detector
    pml_size: int = params_json['pml_size']                                    # PML size   
    pml_alpha: float = params_json['pml_alpha']                                # PML absorption coefficient [Np/grid point]
    CFL: float = params_json['CFL']                                            # CFL number        

    # grid
    dx: float = c0 / (points_per_wavelength * source_freq) 
    Nx: int = wavelength_separation * points_per_wavelength + 20               # number of grid points in the x direction (distance + 20) [grid point]

    # medium
    sound_speed: float = c0
    density: float = rho0
    alpha_power: int = 2
    alpha_coeff: float = alpha_0
    f_max: float = sound_speed / (2 * dx) 
    
    # source
    source_pos: int = 10                                                        # source position [grid point]
    x0: float = (source_pos-1) * dx                                             # source position [m]
    
    # sensor
    x_px: int = wavelength_separation * points_per_wavelength                   # number of grid points between the source and detector [grid point]
    x: float = x_px * dx                                                        # distance between the source and detector [m]
    x_end: float = (x_px + 20) * dx                                             # end of the grid in the x direction [m]

    detector_pos: int = source_pos + x_px                                       # position of the detector [grid point]
    x_detector: float = (detector_pos-1) * dx                                   # position of the detector [m]

    # nonlinearity
    k: float = 2 * pi * source_freq / c0
    BonA: float = 2 * (sigma / (p0 / (rho0 * c0**2) * k * x) - 1)
    tau: float = 0.0
    eta: float = 0.0
    d: float = 0.0
    y: float = 1.0
    
    # time
    dt: float = 1 / (round(points_per_wavelength / CFL) * source_freq)
    t_end: float = 25e-6
    Nt: int = round(t_end / dt)

@dataclass
class NonlinearHomo1D:
    '''based on K-Wave example 'example_na_modelling_nonlinearity.m' '''
    with open('params.json', 'r') as f:
        params_json = json.load(f)['NonlinearHomo1D']

    # parameters
    p0: float = params_json['p0']                                              # source pressure [Pa]
    c0: float = params_json['c0']                                              # sound speed [m/s]
    rho0: float = params_json['rho0']                                          # density [kg/m^3]
    alpha_0: float = params_json['alpha_0']                                    # absorption coefficient [dB/(MHz^2 cm)]
    sigma: float = params_json['sigma']                                        # shock parameter
    source_freq: float = params_json['source_freq']                            # frequency [Hz]
    points_per_wavelength: int = params_json['points_per_wavelength']          # number of grid points per wavelength at f0 [grid point]
    wavelength_separation: int = params_json['wavelength_separation']          # separation (number of waves) between the source and detector
    pml_size: int = params_json['pml_size']                                    # PML size   
    pml_alpha: float = params_json['pml_alpha']                                # PML absorption coefficient [Np/grid point]
    CFL: float = params_json['CFL']                                            # CFL number        

    # grid
    dx: float = c0 / (points_per_wavelength * source_freq)                     # grid point spacing [m]
    Nx: int = wavelength_separation * points_per_wavelength + 20               # number of grid points in the x direction (distance + 20) [grid point]

    # medium
    sound_speed: float = c0
    density: float = rho0
    alpha_power: int = 2
    alpha_coeff: float = alpha_0
    f_max: float = sound_speed / (2 * dx) 
    
    # source
    source_pos: int = 10                                                        # source position [grid point]
    x0: float = (source_pos-1) * dx                                             # source position [m]
    
    # sensor
    x_px: int = wavelength_separation * points_per_wavelength                   # number of grid points between the source and detector [grid point]
    x: float = x_px * dx                                                        # distance between the source and detector [m]
    x_end: float = (x_px + 20) * dx                                             # end of the grid in the x direction [m]

    detector_pos: int = source_pos + x_px                                       # position of the detector [grid point]
    x_detector: float = (detector_pos-1) * dx                                   # position of the detector [m]

    # nonlinearity
    k: float = 2 * pi * source_freq / c0
    BonA: float = 2 * (sigma / (p0 / (rho0 * c0**2) * k * x) - 1)
    y: float = 1.0                                                              # power law exponent
    tau: float = -2 * alpha_0 * c0**(y-1)
    eta: float = 2 * alpha_0 * c0**y * tan(pi*y/2)
    d: float = 0.0                                                              # particle displacement vector
    
    # time
    dt: float = 1 / (round(points_per_wavelength / CFL) * source_freq)
    t_end: float = 25e-6
    Nt: int = round(t_end / dt)

# @dataclass
# class WaveParams1D:
#     '''based on coeficients in the pde system'''
#     # physical parmeters
#     p0: float = 10e6
#     c0: float = 1500.0
#     rho0: float = 1000.0
#     source_freq: float = 1e6
#     alpha_0: float = 1e-6   # absorption coefficient
#     sigma: float = 1e-6     # shock parameter

#     BonA: float = 1.0
#     tau: float = 0.0
#     eta: float = 0.0
#     d: float = 0.0
#     y: float = 1.0
#     Nx: int = 1200
#     Nt: int = 1200
#     t_end: float = 25e-6

# p0: float = 1e-6                     # source pressure [Pa]
# c0: float = 1500                     # sound speed [m/s]
# rho0: float = 1000                   # density [kg/m^3]
# alpha_0: float = 1e-6                # absorption coefficient [dB/(MHz^2 cm)]
# sigma: float = 1e-6                  # shock parameter
# source_freq: float = 1e6             # frequency [Hz]
# points_per_wavelength: int = 3       # number of grid points per wavelength at f0 [grid point]
# wavelength_separation: int = 15      # separation (number of waves) between the source and detector
# pml_size: int = 80                   # PML size
# pml_alpha: float = 1.5               # PML absorption coefficient [Np/grid point]
# CFL: float = 0.25                    # CFL number

